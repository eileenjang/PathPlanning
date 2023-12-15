"""
Cubic spline planner

Author: Atsushi Sakai(@Atsushi_twi)

"""
import math
import numpy as np
import bisect


class Spline:

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []
        self.x = x # s (누적합)
        self.y = y # result_path[:, 0] or result_path[:, 0]
        self.nx = len(x)  # dimension of x
        h = np.diff(x)
        # calc coefficient c
        self.a = [iy for iy in y] # y list.
        # calc coefficient c
        A = self.__calc_A(h) # s값(누적합)과 관련됨.
        B = self.__calc_B(h) # y(x or y)값과 관련됨.

        self.c = np.linalg.solve(A, B) # 연립 방정식을 푼다, result_path와 관련된 무언가 일듯.
        
        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i])) # 무언가
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb) # 무언가

    def calc(self, t): # t : fp.s[i] FrenetPath에서의 종방향 값들의 집합, 여기 s 2개임.

        if t < self.x[0]: # t는 s.s(누적합) 안에 있는 값 -> 경로 내부의 값인지 체크 하는 것
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t) # result_path(오름차순)에 fp.s[i]가 들어갈 자리를 리턴
        dx = t - self.x[i] # 현재 위치(fp.s[i] : 종방향)와 바로 다음 노드의 위치의 차이를 구함.
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        return result # a : y list, b, c, d는 모르겠음. 근데 3차방정식인건 알겠음. but 값이 있어서 float 로 나옴.

    def calcd(self, t):

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result # 이차방정식 리턴 (도함수)

    def calcdd(self, t):

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result # 일차식 (이계도 함수)

    def __search_index(self, x):
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h): # h = np.diff(x) 신기한 행렬.
        A = np.zeros((self.nx, self.nx)) # x의 길이
        A[0, 0] = 1.0
        for i in range(self.nx - 1): # 0행 0렬은 이미 있으니까
            if i != (self.nx - 2): # 마지막이 아닐 때
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h):
        B = np.zeros(self.nx)

        for i in range(self.nx - 2):
            # a : y list
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]

        return B


class Spline2D:

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x) # result_path[:, 0]
        self.sy = Spline(self.s, y) # result_path[:, 1]

    def __calc_s(self, x, y):
        # current - next
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        # 유클리드 거리 계산
        s = [0]
        s.extend(np.cumsum(self.ds)) # 누적되는 점수, 누적 합
        return s

    def calc_position(self, s):
        #여기가 문제
        x = self.sx.calc(s) # x값과 관련된 3차 방정식
        y = self.sy.calc(s) # y값과 관련된 3차 방정식
        return x, y

    def calc_curvature(self, s):
        dx = self.sx.calcd(s)
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s): # input: 종방향 위치, 하나의 상수
        dx = self.sx.calcd(s) # 속도
        dy = self.sy.calcd(s) # 속도
        yaw = math.atan2(dy, dx) # 속도 벡터의 방향
        return yaw


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def main():  # pragma: no cover
    print("Spline 2D test")
    import matplotlib.pyplot as plt
    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    ds = 0.1  # [m] distance of each intepolated points
    calc_spline_course(x,y, ds)
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))
    # plt.subplots(1)
    # plt.plot(x, y, "xb", label="input")
    # plt.plot(rx, ry, "-r", label="spline")
    # plt.grid(True)
    # plt.axis("equal")
    # plt.xlabel("x[m]")
    # plt.ylabel("y[m]")
    # plt.legend()

    # plt.subplots(1)
    # plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    # plt.grid(True)
    # plt.legend()
    # plt.xlabel("line length[m]")
    # plt.ylabel("yaw angle[deg]")

    # plt.subplots(1)
    # plt.plot(s, rk, "-r", label="curvature")
    # plt.grid(True)
    # plt.legend()
    # plt.xlabel("line length[m]")
    # plt.ylabel("curvature [1/m]")

    # plt.show()


if __name__ == '__main__':
    main()
