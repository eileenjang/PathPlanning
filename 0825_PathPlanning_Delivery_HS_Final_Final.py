import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import copy
import math
import pymap3d as pm
import cubic_spline_planner
from quintic_polynomials_planner import QuinticPolynomial
from multiprocessing import Process, Value, Manager
from datetime import datetime
from canlib import canlib, Frame, kvadblib
from canlib.canlib import ChannelData
import time as timee

dbc_llh = kvadblib.Dbc(filename='02_GPS_INS_Parser_1206.dbc')
dbc_end = kvadblib.Dbc(filename='06_Path_0726_final.dbc')

#---------------------------GPS_INS----------------------------------

VEHICLE_INF_2 = dbc_llh.get_message_by_name('INS_VelXYZ')
VEHICLE_INF_2_sig = VEHICLE_INF_2.bind()
# INS_VelXYZ = VEHICLE_INF_2
RTK_GPS_Latitude = dbc_llh.get_message_by_name('RTK_Latitude')
RTK_GPS_Latitude_sig = RTK_GPS_Latitude.bind()

RTK_GPS_Longitude = dbc_llh.get_message_by_name('RTK_Longitude')
RTK_GPS_Longitude_sig = RTK_GPS_Longitude.bind()

#---------------------------GPS_INS----------------------------------
# ---------------------------Path----------------------------------

Wave_Path_Next_LL = dbc_end.get_message_by_name("Wave_Path_Next_LL")
Wave_Path_Next_LL_sig = Wave_Path_Next_LL.bind()

Car_info = dbc_end.get_message_by_name("Car_info")
Car_info_sig = Car_info.bind()

# ---------------------------Path----------------------------------

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 100.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s] # 50.0 / 3.6
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 1.5  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = True

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.counting = 0

def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []
    cnt_dict = {}
    cnt = 0

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                tfp.counting = cnt
                frenet_paths.append(tfp)

                cnt_dict[cnt] = [tfp, lat_qp, lon_qp]
                cnt += 1

    return frenet_paths, cnt_dict

def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    try:
        for i in range(len(ob[:, 0])):
            d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
                 for (ix, iy) in zip(fp.x, fp.y)]

            collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

            if collision:
                return False
    except:
        pass

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            print(1)
            # continue
            pass
        # elif any([abs(a) > MAX_ACCEL for a in
        #           fplist[i].s_dd]):  # Max accel check
        #     print(2)
        #     continue
        # elif any([abs(c) > MAX_CURVATURE for c in
        #           fplist[i].c]):  # Max curvature check
        #     print(3)
        #     continue
        if not check_collision(fplist[i], ob):
            print(4)
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(index, current_course_position, c_speed, lateral_position, lateral_speed, acceleration, obstacle, p_cnt_dict):

    fplist, cnt_dicts = calc_frenet_paths(c_speed, lateral_position, lateral_speed, acceleration, current_course_position)

    # 이거만으로는 초기값을 설명할 수 없음.
    fplist = calc_global_paths(fplist, index)

    fplist = check_paths(fplist, obstacle)

    min_cost = float("inf") # 무한대
    best_path = None
    for fp in fplist:

        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    error_idx = 0

    try:
        num = best_path.counting
        indexing = cnt_dicts[num]
        if c_speed < 1.5:
            return best_path, p_cnt_dict, error_idx
        return best_path, indexing, error_idx  # lat_qp, lon_qp
        # return best_path, indexing[1], indexing[2]  # lat_qp, lon_qp

    except:
        best_path = copy.deepcopy(p_cnt_dict[0])
        if c_speed < 1.5:
            return best_path, p_cnt_dict, error_idx
        del best_path.d[2]
        del best_path.d_d[2]
        del best_path.d_dd[2]
        del best_path.x[2]
        del best_path.y[2]
        error_idx += 1
        print("No best_path, error occured!!")
        return best_path, p_cnt_dict, error_idx

def get_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

def find_channel(ch_num):
    return_channel_num = 0  # 명령창에 CAN Driver 종류가 나올거임
                           # 해당하는 채널 번호를 넣어주면 된다!!
                           # 밑에는 그냥 지금 어떤 채널이 잡히고 있는지 출력하는거임
                           # 밑에 신경쓰지말고 숫자만 제대로 넣으면 된다.

    num = canlib.getNumberOfChannels()

    for channel in range(0, num):
        chdata = canlib.ChannelData(channel)

        print("%d, %s, %s, %s" % (channel, chdata.channel_name, chdata.card_upc_no, chdata.card_serial_no))
        #ean = canlib.getChannelData_EAN(channel)
        ean = ChannelData(channel).card_upc_no # 위아래 2개를 안될때 바꿔가면서 해볼것
        # virtual can 만 연결되었을 때 00000-0이 뜨고 그 외의 경우 00000-0과 다른 무언가가 뜬다. 따라서 00000-0만 있는 경우를 필터링하면 된다.
        # if '00509-9' in ean :   #'00509-9' 는 크바이저 1채널 짜리 # ERP42 - > 00509-9
        #    return_channel_num = channel + ch_num #채널 개수가 0,1,2,3으로 증가함
        #    break
        ean = list(set(ean))
        if len(ean) == 1 and ean[0] == '00000-0':
            return_channel_num = channel + ch_num #채널 개수가 0,1,2,3으로 증가함
            break

    return return_channel_num


def setUpChannel(channel, openFlags=canlib.canOPEN_ACCEPT_VIRTUAL, bitrate=canlib.canBITRATE_500K, bitrateFlags=canlib.canDRIVER_NORMAL):
    ch = canlib.openChannel(channel, openFlags)
    # print("Using channel: %s, EAN: %s" % (ChannelData(channel).device_name,
    #                                       ChannelData(channel).card_upc_no)
    #                                           )
    ch.setBusOutputControl(bitrateFlags)
    ch.setBusParams(bitrate)
    ch.busOn()
    return ch

def div_num(num):
    num_0 = int(num)
    dec_len = len(str(num)) - len(str(num_0)) - 1
    num_dec = str(round(num - num_0, dec_len))

    return num_0, float(num_dec)

def Vel(velocity, current_lat, current_lon, ALL_STOP):
    channel_num = find_channel(1)
    ch1 = setUpChannel(channel_num)

    while True:
        if ALL_STOP.value == 1:
            break
        try:

            frame = ch1.read()
            # INS_VelXYZ = VEHICLE_INF_2
            if frame.id == 0x41c:
                VEHICLE_INF_2_sig = VEHICLE_INF_2.bind(frame)
                # velocity.value = round(VEHICLE_INF_2_sig.RTK_KPH.phys, 4)  # Update my velocity
                vel_x =  round(VEHICLE_INF_2_sig.INS_VelX.phys, 4)
                vel_y =  round(VEHICLE_INF_2_sig.INS_VelY.phys, 4)
                velocity.value = math.sqrt((vel_x ** 2) + (vel_y ** 2))  # Update my velocity

            elif frame.id == 0x401:
                RTK_GPS_Latitude_sig = RTK_GPS_Latitude.bind(frame)

                S_lat_int = int(RTK_GPS_Latitude_sig.RTK_Lat_Int.phys)
                S_lat_double = float(RTK_GPS_Latitude_sig.RTK_Lat_Dec.phys)
                current_lat.value = S_lat_int + (S_lat_double / 100000000)
                print(current_lat.value)

            elif frame.id == 0x402:
                RTK_GPS_Longitude_sig = RTK_GPS_Longitude.bind(frame)
                S_lon_int = int(RTK_GPS_Longitude_sig.RTK_Long_Int.phys)
                S_lon_double = float(RTK_GPS_Longitude_sig.RTK_Long_Dec.phys)
                current_lon.value = S_lon_int + (S_lon_double / 100000000)
                print(current_lon.value)

        except (canlib.canNoMsg) as ex:
            pass
    print("CAR INFORMATION Parshing stop")

    # + test

def pre_Delivery(current_lat, current_lon, car_state, delivery_dest, Delivery_GPP_list, lidar_dict):
    print("pre_Delivery")
    while True:
        if car_state.value == 1:
            return
        
        timee.sleep(20)
        
        ## TODO check delivery_dest with camera
        delivery_dest.value = 0
        car_state.value = 8

def preDelivery_gpp(vel, current_lat, current_lon, current_alt, lidar_dict, light, ALL_STOP, CNT, GPP_list, cur_gpp_pos, car_state, delivery_dest):
    print("preDelivery_gpp")
    channel_num = find_channel(0)# for sending
    ch3 = setUpChannel(channel_num)
    # 업로드 빈도는 0.5초
    matplotlib.use("TKAgg")

    num = int(cur_gpp_pos.value)
    if len(GPP_list) < num:
        print("end!")
        return
    GPP = GPP_list[num]

    result_path = copy.deepcopy(GPP)

    center = sum(GPP) / len(GPP) # 지도의 원점

    ENU_all = []

    for llh in GPP:
        e, n, u = pm.geodetic2enu(llh[0], llh[1], 0, center[0], center[1], 0) # 위도, 경도 순
        ENU_all.append([e, n])
    ENU_all = np.array(ENU_all)

    temp_result = []
    for llh in result_path:
        e, n, u = pm.geodetic2enu(llh[0], llh[1], 0, center[0], center[1], 0)
        temp_result.append([e, n])
    result_path = np.array(temp_result)

    dx, dy, dyaw, dk, s = get_course(result_path[:, 0], result_path[:, 1])


    c_d = 0.0
    c_d_d = 0.0
    c_d_dd = 0.0

    list_dict = {}

    for index, lists in enumerate(result_path):
        list_dict[index] = lists

    now_time = datetime.now()

    comp_x, comp_y, comp_z = pm.geodetic2enu(current_lat.value, current_lon.value, 0, center[0], center[1], 0)
    temp_dict = {}

    for key, value in list_dict.items():
        point_position = np.hypot(value[0] - comp_x, value[1] - comp_y)
        temp_dict[key] = point_position

    POINT = min(temp_dict.keys(), key=lambda k: temp_dict[k])

    obstacle = []
    s0 = s.s[POINT]

    p_indexing = [FrenetPath()]
    velocity0 = vel.value

    path, indexing, error = frenet_optimal_planning(s, s0, velocity0, c_d, c_d_d, c_d_dd, obstacle, p_indexing)

    try:
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
    except:
        c_d = 0
        c_d_d = 0
        c_d_dd = 0
        pass

    p_indexing = indexing

    next_lat, next_lon, next_alt = pm.enu2geodetic(path.x[1], path.y[1], 0, center[0],
                                                   center[1], 0)

    lon_prev = next_lon
    lat_prev = next_lat

    time0 = timee.time()

    pos_x = []
    pos_y = []

    res_x = []
    res_y = []

    while True:
        if ALL_STOP.value == 1:
            break

        if car_state.value == 6 or car_state.value == 7 or car_state.value == 8:
            pass

        else:
            plt.close()
            car_state.value == 0
            return

        comp_x, comp_y, comp_z = pm.geodetic2enu(current_lat.value, current_lon.value, 0, center[0], center[1], 0)

        plot_x = copy.deepcopy(comp_x)
        plot_y = copy.deepcopy(comp_y)
        temp_dict = {}

        velocity0 = vel.value

        for key, value in list_dict.items():
            point_position = np.hypot(value[0] - comp_x, value[1] - comp_y)
            temp_dict[key] = point_position

        POINT = min(temp_dict.keys(), key=lambda k: temp_dict[k])

        s0 = s.s[POINT]

        obstacle = []
        end_time = datetime.now()
        time = (end_time - now_time).seconds
        path, indexing, error = frenet_optimal_planning(s, s0, velocity0, c_d, c_d_d, c_d_dd, obstacle, p_indexing)

        try:
            c_d = path.d[1]
            c_d_d = path.d_d[1]
            c_d_dd = path.d_dd[1]
        except:
            c_d = 0
            c_d_d = 0
            c_d_dd = 0
            pass

        break_distance = round(0.0005 * math.pow(velocity0 * 3.6, 2) + 0.2 * (velocity0 * 3.6), 3)

        lookahead = velocity0 * 0.1 + 3.0


        if error != 1:

            x_path = copy.deepcopy(path.x[:2])
            y_path = copy.deepcopy(path.y[:2])

            for i in range(2, len(path.x)):
                distance = np.hypot(path.x[i] - path.x[1], path.y[i] - path.y[1])
                dists = distance - lookahead
                if dists > 0:
                    x_path.append(path.x[i])
                    y_path.append(path.y[i])

            path.x = x_path
            path.y = y_path


        next_lat, next_lon, next_alt = pm.enu2geodetic(path.x[2], path.y[2], 0, center[0],
                                                           center[1], 0)

        comp_end = np.hypot(path.x[1] - dx[-1], path.y[1] - dy[-1])

        try:
            runtime = timee.time() - time0

            print('runtime : {}'.format(runtime))

            if runtime >= 0.5:
                time0 = timee.time()  # 시간 측정 시작

                canlib.IOControl(ch3).flush_tx_buffer()
                # --------------------------------------NEXT---------------------------------
                WPNLat_0, WPNLat_1 = div_num(next_lat)

                WPNLon_0, WPNLon_1 = div_num(next_lon)

                Wave_Path_Next_LL_sig.Wave_Path_Next_Lat_Int.phys = WPNLat_0

                Wave_Path_Next_LL_sig.Wave_Path_Next_Lat_Dec.phys = WPNLat_1 * 100_000_000

                Wave_Path_Next_LL_sig.Wave_Path_Next_Long_Int.phys = WPNLon_0

                Wave_Path_Next_LL_sig.Wave_Path_Next_Long_Dec.phys = WPNLon_1 * 10_000_000

                Wave_Path_Next_LL_sig.Wave_Path_Next_CNT.phys = CNT.value

                Wave_Path_Next_LL_sig.Wave_Path_Next_CTC.phys = 1

                ch3.write(Wave_Path_Next_LL_sig._frame)


                # --------------------------------------CNT, CTC--------------------------------

                Car_info_sig.ALL_STOP.phys = ALL_STOP.value

                Car_info_sig.Car_state.phys = car_state.value

                Car_info_sig.Car_info_CNT.phys = CNT.value

                Car_info_sig.Car_info_CTC.phys = 1

                Car_info_sig.Delivery_dest.phys = delivery_dest.value

                Car_info_sig.AEB_sign.phys = 0

                ch3.write(Car_info_sig._frame)

                lon_prev = next_lon
                lat_prev = next_lat

                CNT.value += 1

            else:

                canlib.IOControl(ch3).flush_tx_buffer()
                # --------------------------------------NEXT---------------------------------

                WPNLat_0, WPNLat_1 = div_num(next_lat)

                WPNLon_0, WPNLon_1 = div_num(next_lon)

                Wave_Path_Next_LL_sig.Wave_Path_Next_Lat_Int.phys = WPNLat_0

                Wave_Path_Next_LL_sig.Wave_Path_Next_Lat_Dec.phys = WPNLat_1 * 100_000_000

                Wave_Path_Next_LL_sig.Wave_Path_Next_Long_Int.phys = WPNLon_0

                Wave_Path_Next_LL_sig.Wave_Path_Next_Long_Dec.phys = WPNLon_1 * 10_000_000

                Wave_Path_Next_LL_sig.Wave_Path_Next_CNT.phys = CNT.value

                Wave_Path_Next_LL_sig.Wave_Path_Next_CTC.phys = 1

                ch3.write(Wave_Path_Next_LL_sig._frame)


                # --------------------------------------CNT, CTC--------------------------------

                Car_info_sig.ALL_STOP.phys = ALL_STOP.value

                Car_info_sig.Car_state.phys = car_state.value

                Car_info_sig.Car_info_CNT.phys = CNT.value

                Car_info_sig.Car_info_CTC.phys = 1

                Car_info_sig.Delivery_dest.phys = delivery_dest.value

                Car_info_sig.AEB_sign.phys = 0

                ch3.write(Car_info_sig._frame)

                lon_prev = next_lon
                lat_prev = next_lat

                CNT.value += 1

        except (canlib.canNoMsg) as ex:
            pass

        if comp_end <= break_distance: # 배달미션 종료
            cur_gpp_pos.value += 1
            car_state.value = 6
            delivery_dest.value = 1 ###여기 값 수정해야해!!
            break

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            pos_x.append(plot_x)
            pos_y.append(plot_y)

            res_x.append(path.x[2])
            res_y.append(path.y[2])

            # line of the world
            plt.plot(ENU_all[:, 0], ENU_all[:, 1], '*k')
            # line of the GPP
            plt.plot(dx, dy, '--k', linewidth=3)
            if len(obstacle) > 0:
                plt.plot(obstacle[:, 0], obstacle[:, 1], "or", markersize=6)

            try:
                # # line of the LPP
                plt.plot(comp_x, comp_y, "or", markersize=10)
                # plt.plot(path.x[1:], path.y[1:], "-og", linewidth=2)
                plt.plot(path.x[1], path.y[1], "vc", markersize=10)
                plt.plot(pos_x, pos_y, "-or", linewidth=1)
                plt.plot(res_x, res_y, "-og", linewidth=2)
                # plt.plot(traffic_all[:, 0], traffic_all[:, 1], '*y')
                # plt.plot(traffic_all[POINT][0], traffic_all[POINT][1], 'Xm', markersize=10)
                # ROI
                # plt.ylim(path.y[1] - 20, path.y[1] + 20)  # -5. + 8
                # plt.xlim(path.x[1] - 20, path.x[1] + 20)  # - 15, +15

                plt.ylim(comp_y - 20, comp_y + 20)  # - 15, +15
                plt.xlim(comp_x - 20, comp_x + 20)  # - 15, +15

                # plt.ylim(plot_y - 2000, plot_y + 2000)  # -5. + 8
                # plt.xlim(plot_x - 2000, plot_x + 2000)  # - 15, +15

                # plt.xlim(path.x[1] - area, path.x[1] + area)
                # plt.ylim(path.y[1] - area, path.y[1] + area)
            except:
                pass

            text = "Time: " + str(time) + " / Velocity: " + str(round(velocity0 * 3.6, 2)) + "km/h"
            plt.title(text)
            plt.grid(True)
            plt.pause(0.0001)

def gpp2lpp(vel, current_lat, current_lon, current_alt, lidar_dict, light, ALL_STOP, CNT, GPP_list, car_state, cur_gpp_pos):
    print("gpp2lpp")
    channel_num = find_channel(0)# for sending
    ch3 = setUpChannel(channel_num)
    # 업로드 빈도는 0.5초
    matplotlib.use("TKAgg")

    # with open("location.txt", "r") as f:


    num = int(cur_gpp_pos.value)
    if len(GPP_list) < num:
        print("end!")
        return
    GPP = GPP_list[num]
    result_path = copy.deepcopy(GPP)


    # obstacle = [[37.45066331,126.6513738]]
    obstacle = [[0,0]]
    obstacle = np.array(obstacle)
    plt.title("GPP")
    # plt.plot(Coordination[:, 1], Coordination[:, 0], "*b", label="HD Map")
    plt.plot(result_path[:, 1], result_path[:, 0], "--r", linewidth=3, label="GPP")
    plt.plot(result_path[0, 1], result_path[0, 0], "Xm", markersize=10, label="Start Point")
    plt.plot(result_path[-1, 1], result_path[-1, 0], "Xg", markersize=10, label="End Point")
    # plt.plot(obstacle[:, 1], obstacle[:, 0], "or", markersize=6)
    plt.legend()
    plt.show()

    center = sum(GPP) / len(GPP) # 지도의 원점

    ENU_all = []

    for llh in GPP:
        e, n, u = pm.geodetic2enu(llh[0], llh[1], 0, center[0], center[1], 0) # 위도, 경도 순
        ENU_all.append([e, n])
    ENU_all = np.array(ENU_all)

    temp_result = []
    for llh in result_path:
        e, n, u = pm.geodetic2enu(llh[0], llh[1], 0, center[0], center[1], 0)
        temp_result.append([e, n])
    result_path = np.array(temp_result)

    dx, dy, dyaw, dk, s = get_course(result_path[:, 0], result_path[:, 1])


    c_d = 0.0
    c_d_d = 0.0
    c_d_dd = 0.0

    list_dict = {}

    for index, lists in enumerate(result_path):
        list_dict[index] = lists

    now_time = datetime.now()

    comp_x, comp_y, comp_z = pm.geodetic2enu(current_lat.value, current_lon.value, 0, center[0], center[1], 0)
    temp_dict = {}

    for key, value in list_dict.items():
        point_position = np.hypot(value[0] - comp_x, value[1] - comp_y)
        temp_dict[key] = point_position

    POINT = min(temp_dict.keys(), key=lambda k: temp_dict[k])

    s0 = s.s[POINT]

    obstacle = []

    ex_obstacle = [[0,0]]
    ex_obstacle = np.array(ex_obstacle)

    for i in range(len(ex_obstacle)):
        e, n, u = pm.geodetic2enu(ex_obstacle[i][0], ex_obstacle[i][1], 0, center[0], center[1], 0)
        obstacle.append([e, n])

    obstacle = np.array(obstacle)
    p_indexing = [FrenetPath()]
    velocity0 = vel.value

    path, indexing, error = frenet_optimal_planning(s, s0, velocity0, c_d, c_d_d, c_d_dd, obstacle, p_indexing)

    try:
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
    except:
        c_d = 0
        c_d_d = 0
        c_d_dd = 0
        pass

    p_indexing = indexing

    next_lat, next_lon, next_alt = pm.enu2geodetic(path.x[1], path.y[1], 0, center[0],
                                                   center[1], 0)

    lon_prev = next_lon
    lat_prev = next_lat

    time0 = timee.time()

    pos_x = []
    pos_y = []

    res_x = []
    res_y = []

    while True:
        if ALL_STOP.value == 1:
            break

        comp_x, comp_y, comp_z = pm.geodetic2enu(current_lat.value, current_lon.value, 0, center[0], center[1], 0)

        plot_x = copy.deepcopy(comp_x)
        plot_y = copy.deepcopy(comp_y)
        temp_dict = {}

        velocity0 = vel.value

        for key, value in list_dict.items():
            point_position = np.hypot(value[0] - comp_x, value[1] - comp_y)
            temp_dict[key] = point_position

        POINT = min(temp_dict.keys(), key=lambda k: temp_dict[k])

        s0 = s.s[POINT]

        obstacle = []

        ex_obstacle = [[37.45066667,126.6514999]]
        ex_obstacle = np.array(ex_obstacle)

        for i in range(len(ex_obstacle)):
            e, n, u = pm.geodetic2enu(ex_obstacle[i][0], ex_obstacle[i][1], 0, center[0], center[1], 0)
            obstacle.append([e, n])

        end_time = datetime.now()
        time = (end_time - now_time).seconds

        obstacle = np.array(obstacle)

        path, indexing, error = frenet_optimal_planning(s, s0, velocity0, c_d, c_d_d, c_d_dd, obstacle, p_indexing)

        try:
            c_d = path.d[1]
            c_d_d = path.d_d[1]
            c_d_dd = path.d_dd[1]
        except:
            c_d = 0
            c_d_d = 0
            c_d_dd = 0
            pass

        break_distance = round(0.0005 * math.pow(velocity0 * 3.6, 2) + 0.2 * (velocity0 * 3.6), 3)

        lookahead = velocity0 * 0.1 + 2.5

        if error != 1:

            x_path = copy.deepcopy(path.x[:2])
            y_path = copy.deepcopy(path.y[:2])

            for i in range(2, len(path.x)):
                distance = np.hypot(path.x[i] - path.x[1], path.y[i] - path.y[1])
                dists = distance - lookahead
                if dists > 0:
                    x_path.append(path.x[i])
                    y_path.append(path.y[i])

            path.x = x_path
            path.y = y_path


        next_lat, next_lon, next_alt = pm.enu2geodetic(path.x[2], path.y[2], 0, center[0],
                                                           center[1], 0)

        comp_end = np.hypot(path.x[1] - dx[-1], path.y[1] - dy[-1])

        try:
            runtime = timee.time() - time0

            print('runtime : {}'.format(runtime))

            if runtime >= 0.5:
                time0 = timee.time()  # 시간 측정 시작

                canlib.IOControl(ch3).flush_tx_buffer()
                # --------------------------------------NEXT---------------------------------
                WPNLat_0, WPNLat_1 = div_num(next_lat)

                WPNLon_0, WPNLon_1 = div_num(next_lon)

                Wave_Path_Next_LL_sig.Wave_Path_Next_Lat_Int.phys = WPNLat_0

                Wave_Path_Next_LL_sig.Wave_Path_Next_Lat_Dec.phys = WPNLat_1 * 100_000_000

                Wave_Path_Next_LL_sig.Wave_Path_Next_Long_Int.phys = WPNLon_0

                Wave_Path_Next_LL_sig.Wave_Path_Next_Long_Dec.phys = WPNLon_1 * 10_000_000

                Wave_Path_Next_LL_sig.Wave_Path_Next_CNT.phys = CNT.value

                Wave_Path_Next_LL_sig.Wave_Path_Next_CTC.phys = 1

                ch3.write(Wave_Path_Next_LL_sig._frame)



                # --------------------------------------CNT, CTC--------------------------------

                Car_info_sig.ALL_STOP.phys = ALL_STOP.value

                Car_info_sig.Car_info_CNT.phys = CNT.value

                Car_info_sig.Car_info_CTC.phys = 1

                Car_info_sig.Parking_pos.phys = 5

                Car_info_sig.AEB_sign.phys = 0

                Car_info_sig.Car_state.phys = car_state.value

                ch3.write(Car_info_sig._frame)

                lon_prev = next_lon
                lat_prev = next_lat

                CNT.value += 1

            else:

                canlib.IOControl(ch3).flush_tx_buffer()
                # --------------------------------------NEXT---------------------------------

                WPNLat_0, WPNLat_1 = div_num(next_lat)

                WPNLon_0, WPNLon_1 = div_num(next_lon)

                Wave_Path_Next_LL_sig.Wave_Path_Next_Lat_Int.phys = WPNLat_0

                Wave_Path_Next_LL_sig.Wave_Path_Next_Lat_Dec.phys = WPNLat_1 * 100_000_000

                Wave_Path_Next_LL_sig.Wave_Path_Next_Long_Int.phys = WPNLon_0

                Wave_Path_Next_LL_sig.Wave_Path_Next_Long_Dec.phys = WPNLon_1 * 10_000_000

                Wave_Path_Next_LL_sig.Wave_Path_Next_CNT.phys = CNT.value

                Wave_Path_Next_LL_sig.Wave_Path_Next_CTC.phys = 1

                ch3.write(Wave_Path_Next_LL_sig._frame)



                # --------------------------------------CNT, CTC--------------------------------

                Car_info_sig.ALL_STOP.phys = ALL_STOP.value

                Car_info_sig.Car_info_CNT.phys = CNT.value

                Car_info_sig.Car_info_CTC.phys = 1

                Car_info_sig.Parking_pos.phys = 5

                Car_info_sig.AEB_sign.phys = 0

                Car_info_sig.Car_state.phys = car_state.value

                ch3.write(Car_info_sig._frame)

                lon_prev = next_lon
                lat_prev = next_lat

                CNT.value += 1

        except (canlib.canNoMsg) as ex:
            pass

        if comp_end <= break_distance:
            print("Goal!")
            ALL_STOP.value = 1
            cur_gpp_pos.value += 1
            car_state.value = 6  #pick up & delivery시작
            break
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            pos_x.append(plot_x)
            pos_y.append(plot_y)

            res_x.append(path.x[2])
            res_y.append(path.y[2])

            # line of the world
            plt.plot(ENU_all[:, 0], ENU_all[:, 1], '*k')
            # line of the GPP
            plt.plot(dx, dy, '--k', linewidth=3)
            if len(obstacle) > 0:
                plt.plot(obstacle[:, 0], obstacle[:, 1], "or", markersize=6)

            try:
                # # line of the LPP
                plt.plot(comp_x, comp_y, "or", markersize=10)
                # plt.plot(path.x[1:], path.y[1:], "-og", linewidth=2)
                plt.plot(path.x[1], path.y[1], "vc", markersize=10)
                plt.plot(pos_x, pos_y, "-or", linewidth=1)
                plt.plot(res_x, res_y, "-og", linewidth=2)
                # plt.plot(traffic_all[:, 0], traffic_all[:, 1], '*y')
                # plt.plot(traffic_all[POINT][0], traffic_all[POINT][1], 'Xm', markersize=10)
                # ROI
                # plt.ylim(path.y[1] - 20, path.y[1] + 20)  # -5. + 8
                # plt.xlim(path.x[1] - 20, path.x[1] + 20)  # - 15, +15

                plt.ylim(comp_y - 20, comp_y + 20)  # - 15, +15
                plt.xlim(comp_x - 20, comp_x + 20)  # - 15, +15

                # plt.ylim(plot_y - 2000, plot_y + 2000)  # -5. + 8
                # plt.xlim(plot_x - 2000, plot_x + 2000)  # - 15, +15

                # plt.xlim(path.x[1] - area, path.x[1] + area)
                # plt.ylim(path.y[1] - area, path.y[1] + area)
            except:
                pass

            text = "Time: " + str(time) + " / Velocity: " + str(round(velocity0 * 3.6, 2)) + "km/h"
            plt.title(text)
            plt.grid(True)
            plt.pause(0.0001)

if __name__ == "__main__":
    ALL_STOP = Value('i', 0)
    velocity = Value('d', 0.0)
    current_lat = Value('d', 0.0)
    current_lon = Value('d', 0.0)
    light = Value('i', 0)
    traffic_sign = Value('i', 0)  # 표지판
    CNT = Value('i', 0)
    cur_gpp_pos = Value('i', 0)
    cur_gpp_pos_main = Value('i', 0)

    gpp2lpp_state = Value('i', 0)
    delivery_dest = Value('i', 0)
    car_state = Value('i', 0) # ex) 0 -> gpp2lpp 실행, 1 -> gpp2lpp 실행 중. 2 -> gpp2lpp가 정상적으로 종료되었다.

    #################################pickup_gpp#################################

    with open("0806_parking_5.csv", "r") as f: # 먼저 enu좌표계로 변환할 것.
#    with open("pickup_gpp.csv", "r") as f: # 먼저 enu좌표계로 변환할 것.
        lines = [i.rstrip("\n") for i in f.readlines()]
        GPP_temp = []
        cnt = 0
        for i in lines:
            if cnt % 4 == 0:
                print(cnt)
                temp = [float(j) for j in i.split(",")]
                GPP_temp.append(temp)
            cnt += 1
        pickup_GPP = np.array(GPP_temp)

    #################################pickup_gpp#################################
    
    #################################delivery_gpp#################################

    with open("0806_parking_1.csv", "r") as f: # 먼저 enu좌표계로 변환할 것.
#    with open("Delivery_1.csv", "r") as f: # 먼저 enu좌표계로 변환할 것.


        lines = [i.rstrip("\n") for i in f.readlines()]
        GPP_temp = []
        cnt = 0
        for i in lines:
            if cnt % 4 == 0:
                print(cnt)
                temp = [float(j) for j in i.split(",")]
                GPP_temp.append(temp)
            cnt += 1
        GPP_1 = np.array(GPP_temp)


    Delivery_GPP_list = [pickup_GPP, GPP_1]

    #################################delivery_gpp#################################

    #################################main_gpp#################################

    with open("0806_parking_5.csv", "r") as f: # 먼저 enu좌표계로 변환할 것.
#    with open("0810_preParking_gpp.csv", "r") as f: # 먼저 enu좌표계로 변환할 것.
        lines = [i.rstrip("\n") for i in f.readlines()]
        GPP_temp = []
        cnt = 0
        for i in lines:
            if cnt % 4 == 0:
                print(cnt)
                temp = [float(j) for j in i.split(",")]
                GPP_temp.append(temp)
            cnt += 1
        main_GPP_1 = np.array(GPP_temp)

    with open("0806_parking_5.csv", "r") as f:  # 먼저 enu좌표계로 변환할 것.
        #    with open("0810_preParking_gpp.csv", "r") as f: # 먼저 enu좌표계로 변환할 것.
        lines = [i.rstrip("\n") for i in f.readlines()]
        GPP_temp = []
        cnt = 0
        for i in lines:
            if cnt % 4 == 0:
                print(cnt)
                temp = [float(j) for j in i.split(",")]
                GPP_temp.append(temp)
            cnt += 1
        main_GPP_2 = np.array(GPP_temp)

        Main_GPP_List = [main_GPP_1, main_GPP_2]

    #################################main_gpp#################################

    V = Process(target=Vel, args=(velocity, current_lat, current_lon, ALL_STOP))
    V.start()

    GPP = Process(target=gpp2lpp, args=(velocity, current_lat, current_lon, 0, 0, light, ALL_STOP, CNT, Main_GPP_List, car_state))
    P_D = Process(target=pre_Delivery, args=(current_lat, current_lon, car_state, delivery_dest, Delivery_GPP_list, 0))
    P_D_G = Process(target=preDelivery_gpp, args=(velocity, current_lat, current_lon, 0, 0, light, ALL_STOP, CNT, Delivery_GPP_list, cur_gpp_pos, car_state, delivery_dest))

    while True:
        # -----------------------------검증용 변수들-------------------------

        # 자동차의 기동에 영향을 미치는 process


        if car_state.value == 6:

            P_D_G = Process(target=preDelivery_gpp, args=(
            velocity, current_lat, current_lon, 0, 0, light, ALL_STOP, CNT, Delivery_GPP_list, cur_gpp_pos, car_state,
            delivery_dest))

            P_D_G.start()
            P_D = Process(target=pre_Delivery,
                          args=(current_lat, current_lon, car_state, delivery_dest, Delivery_GPP_list, 0))

            P_D.start()
            
            car_state.value = 7  # pre_delivery 진행중
            


        elif car_state.value == 0:

            GPP = Process(target=gpp2lpp,
                          args=(velocity, current_lat, current_lon, 0, 0, light, ALL_STOP, CNT, Main_GPP_List, car_state, cur_gpp_pos_main))
            GPP.start()
            car_state.value = 1
            