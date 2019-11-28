import cv2
import numpy as np
import os
from scipy import ndimage


# Octree를 만들기 위한 Node Class 설정
class Node(object):
    def __init__(self):

        # 각 노드가 가지는 속성 키 상태값
        self.key = 0
        self.State = 0

        # 트리구조를 만들기 위해 필요한 요소 ( 부모노드. 자식노드 )
        self.father_node = None

        self.Oc_1 = self.Oc_2 = self.Oc_3 = self.Oc_4 = self.Oc_5 = self.Oc_6 = self.Oc_7 = self.Oc_8 = None


class octree_for_SR(object):
    def __init__(self, data, data_type, Max_Matrix, patchsize, margin, axis, index):
        self.root = None

        self.root_data = data
        self.data_type = data_type
        self.margin = margin
        self.margin_axis = axis
        self.index = index

        self.node_list = []
        self.present_node_list = []

        self.patchsize = patchsize

        self.threshold_min = 0.01

        # Overlap = 데이터를 겹쳐서 Super-resolution 한 뒤에 겹친부분 삭제하면 Artifact가 사라지는 효과 있음
        # TempoGAN 논문 확인
        self.overlap_size = 3

        self.width = len(data[1])
        self.height = len(data[0])
        self.depth = len(data)

        # Empty = Super-resolution 을 할 필요 없는 공간
        # S_SRR = Super-resolution 을 해야 할 공간
        # MIX = SRR과 Empty가 섞여 있는 공간
        self.S_Empty = "Empty"
        self.S_SRR = "S_SRR"
        self.S_Mix = "Mix"

        self.max_patch_size = 32 # OcTree 최상위 노드의 데이터 크기 제한(GPU 메모리 문제)

        self.Max_Matrix = Max_Matrix

        # 최하단 노드의 Patch 개수
        self.terminal_patch_num_x = self.width // self.patchsize
        self.terminal_patch_num_y = self.height // self.patchsize
        self.terminal_patch_num_z = self.depth // self.patchsize

        self.terminal_depth = 0

        self.present_patch_num_x = self.terminal_patch_num_x
        self.present_patch_num_y = self.terminal_patch_num_y
        self.present_patch_num_z = self.terminal_patch_num_z
        self.present_depth = 0

        self.terminal_SRR_key = []
        self.terminal_SRR_data = []

        self.depth0_SRR_key = []
        self.depth1_SRR_key = []
        self.depth2_SRR_key = []
        self.depth3_SRR_key = []
        self.depth4_SRR_key = []

        self.depth0_SRR_data = []
        self.depth1_SRR_data = []
        self.depth2_SRR_data = []
        self.depth3_SRR_data = []
        self.depth4_SRR_data = []

        self.run()

    # 최하단 노드의 깊이 값 계산
    def terminal_depth_compute(self, width, patchsize):
        tmp = width // patchsize
        i = 0
        while True:
            if tmp == pow(2, i):
                self.terminal_depth = i
                break
            i = i + 1

    def terminal_state_check_GPU(self, z, y, x):
        biggest_data_value = self.Max_Matrix[z, y, x]
        # Patch에 해당하는 최대값이 threshold 보다 작으면 비어있는 공간(Super-resolution이 필요 없는 공간으로 상태 설정)
        # threshold 보다 크면 Super-resoluiton을 해야하는 공간으로 상태 설정
        if self.threshold_min > biggest_data_value:
            State = self.S_Empty
        else:
            State = self.S_SRR

        return State

    # Octree를 만들기 위해 최하단 노드 생성
    def Set_terminal_node_GPU(self):
        self.terminal_depth_compute(self.width, self.patchsize) # 깊이 계산

        # 모든 패치를 돌아가며 노드 생성 및 설정
        for k in range(self.terminal_patch_num_z):
            node_list_y = []
            for i in range(self.terminal_patch_num_y):
                node_list_x = []
                for j in range(self.terminal_patch_num_x):
                    # 최하단 노드에 키 상태 넣어주기
                    node = Node()
                    key = (self.terminal_depth, j+1, i+1, k+1)
                    State = self.terminal_state_check_GPU(k, i, j)
                    node.key = key
                    node.State = State
                    node_list_x.append(node)
                node_list_y.append(node_list_x)
            # 노드 리스트에 추가
            self.node_list.append(node_list_y)
        self.present_depth = self.terminal_depth
        self.present_node_list = self.node_list

    # Octree를 만들 때, 자식 노드 8개의 상태값을 보고 부모 노드 상태값 결정 및 데이터 병합
    def Neighbor_State_Search(self, Grid_x, Grid_y, Grid_z):
        # 자식 노드 8개 상태 확인
        xyzstate = self.present_node_list[Grid_z - 1][Grid_y - 1][Grid_x - 1].State
        xyz_state = self.present_node_list[Grid_z - 2][Grid_y - 1][Grid_x - 1].State
        xy_zstate = self.present_node_list[Grid_z - 1][Grid_y - 2][Grid_x - 1].State
        xy_z_state = self.present_node_list[Grid_z - 2][Grid_y - 2][Grid_x - 1].State
        x_yzstate = self.present_node_list[Grid_z - 1][Grid_y - 1][Grid_x - 2].State
        x_yz_state = self.present_node_list[Grid_z - 2][Grid_y - 1][Grid_x - 2].State
        x_y_zstate = self.present_node_list[Grid_z - 1][Grid_y - 2][Grid_x - 2].State
        x_y_z_state = self.present_node_list[Grid_z - 2][Grid_y - 2][Grid_x - 2].State

        # 자식 노드 상태 확인하여 부모노드 상태값 및 데이터 리턴
        if xyzstate == xyz_state == xy_zstate == xy_z_state == x_yzstate == x_yz_state == x_y_zstate == x_y_z_state \
                == self.S_SRR:
            return self.S_SRR
        elif xyzstate == xyz_state == xy_zstate == xy_z_state == x_yzstate == x_yz_state == x_y_zstate == x_y_z_state \
                == self.S_Empty:
            return self.S_Empty
        else:
            return self.S_Mix

    # 부모 노드와 자식 노드 연결
    def Connect_Octree(self, node, Grid_x, Grid_y, Grid_z):
        node.Oc_1 = self.present_node_list[Grid_z - 2][Grid_y - 2][Grid_x - 2]
        node.Oc_2 = self.present_node_list[Grid_z - 2][Grid_y - 2][Grid_x - 1]
        node.Oc_3 = self.present_node_list[Grid_z - 2][Grid_y - 1][Grid_x - 2]
        node.Oc_4 = self.present_node_list[Grid_z - 2][Grid_y - 1][Grid_x - 1]

        node.Oc_5 = self.present_node_list[Grid_z - 1][Grid_y - 2][Grid_x - 2]
        node.Oc_6 = self.present_node_list[Grid_z - 1][Grid_y - 2][Grid_x - 1]
        node.Oc_7 = self.present_node_list[Grid_z - 1][Grid_y - 1][Grid_x - 2]
        node.Oc_8 = self.present_node_list[Grid_z - 1][Grid_y - 1][Grid_x - 1]
        return node

    # 트리 생성 함수
    def Set_tree(self, node):
        # 부모 노드가 최상위 노드일 때
        if self.present_depth == 1:
            Grid_x = self.present_patch_num_x
            Grid_y = self.present_patch_num_y
            Grid_z = self.present_patch_num_z
            self.present_depth = self.present_depth - 1
            for z in range(2, Grid_z + 1, 2):
                for y in range(2, Grid_y + 1, 2):
                    for x in range(2, Grid_x + 1, 2):
                        # 부모 노드의 키 상태값 계산해서 부모노드 생성 및 설정
                        root_State = self.Neighbor_State_Search(x, y, z)
                        node = Node()
                        self.present_node_list[z - 1][y - 1][x - 1].father_node = node
                        key = (self.present_depth, x - 1, y - 1, z - 1)

                        node.key = key
                        node.State = root_State
                        self.Connect_Octree(node, x, y, z)
            self.root = node
        # 부모 노드가 최상위 노드가 아니면
        else:
            Grid_x = self.present_patch_num_x
            Grid_y = self.present_patch_num_y
            Grid_z = self.present_patch_num_z
            self.present_depth = self.present_depth - 1
            self.present_patch_num_x = self.present_patch_num_x // 2
            self.present_patch_num_y = self.present_patch_num_y // 2
            self.present_patch_num_z = self.present_patch_num_z // 2
            node_list = []
            for z in range(2, Grid_z + 1, 2):
                node_list_y = []
                for y in range(2, Grid_y + 1, 2):
                    node_list_x = []
                    for x in range(2, Grid_x + 1, 2):
                        # 부모 노드의 키 상태값 계산해서 부모노드 생성 및 설정
                        father_State = self.Neighbor_State_Search(x, y, z)
                        node = Node()
                        self.present_node_list[z - 1][y - 1][x - 1].father_node = node
                        key = (self.present_depth, x // 2, y // 2, z // 2)

                        node.key = key
                        node.State = father_State
                        node_list_x.append(node)
                    node_list_y.append(node_list_x)
                node_list.append(node_list_y)

            for z in range(2, Grid_z + 1, 2):
                for y in range(2, Grid_y + 1, 2):
                    for x in range(2, Grid_x + 1, 2):
                        node = self.present_node_list[z - 1][y - 1][x - 1].father_node
                        self.Connect_Octree(node, x, y, z)

            self.present_node_list = node_list

            # 재귀 함수로 돌면서 Octree 생성
            self.Set_tree(self.root)

        return node


    # Overlap 영역까지 Data 분할하여 Super-resolution 해야 할 데이터 리턴
    def node_data_setting_overlap(self, key):
        overlap_size = self.overlap_size

        side_size = self.width // pow(2, key[0])
        # 키의 0번째가 0이면 깊이가 0이라는 뜻 이므로, 깊이가 0이면 Root Node 이다.
        # Root Node Super-resolution은 overlap 불필요
        x = key[1]
        y = key[2]
        z = key[3]

        x_start = (x - 1) * side_size - overlap_size
        y_start = (y - 1) * side_size - overlap_size
        z_start = (z - 1) * side_size - overlap_size

        x_end = x * side_size + overlap_size
        y_end = y * side_size + overlap_size
        z_end = z * side_size + overlap_size

        margin_data = np.zeros(1)

        if x == 1:
            if len(self.margin) > 0 and self.margin_axis == 2 and self.index == 1:
                x_start = 0
            else:
                x_start = 0
                x_end = x * side_size + overlap_size * 2
        if y == 1:
            if len(self.margin) > 0 and self.margin_axis == 1 and self.index == 1:
                y_start = 0
            else:
                y_start = 0
                y_end = y * side_size + overlap_size * 2
        if z == 1:
            if len(self.margin) > 0 and self.margin_axis == 0 and self.index == 1:
                z_start = 0
            else:
                z_start = 0
                z_end = z * side_size + overlap_size * 2

        if x * side_size + overlap_size > self.width:
            if len(self.margin) > 0 and self.margin_axis == 2 and self.index == 0:
                x_end = self.width
            else:
                x_start = (x - 1) * side_size - overlap_size * 2
                x_end = self.width
        if y * side_size + overlap_size > self.width:
            if len(self.margin) > 0 and self.margin_axis == 1 and self.index == 0:
                y_end = self.height
            else:
                y_start = (y - 1) * side_size - overlap_size * 2
                y_end = self.height
        if z * side_size + overlap_size > self.width:
            if len(self.margin) > 0 and self.margin_axis == 0 and self.index == 0:
                z_end = self.depth
            else:
                z_start = (z - 1) * side_size - overlap_size * 2
                z_end = self.depth

        if len(self.margin) > 0 \
                and ((x * side_size + overlap_size > self.width and self.margin_axis == 2 and self.index == 0)
                     or (y * side_size + overlap_size > self.height and self.margin_axis == 1 and self.index == 0)
                     or (z * side_size + overlap_size > self.depth and self.margin_axis == 0 and self.index == 0)):
            data = self.root_data[z_start: z_end, y_start: y_end, x_start: x_end, :]
            margin_data = np.zeros(1)
            if x * side_size + overlap_size > self.width and self.margin_axis == 2:
                margin_data = self.margin[z_start: z_end, y_start: y_end, :, :]
            elif y * side_size + overlap_size > self.height and self.margin_axis == 1:
                margin_data = self.margin[z_start: z_end, :, x_start: x_end, :]
            elif z * side_size + overlap_size > self.depth and self.margin_axis == 0:
                margin_data = self.margin[:, y_start: y_end, x_start: x_end, :]
            data = np.concatenate([data, margin_data], axis=self.margin_axis)
        elif len(self.margin) > 0 \
                and ((x == 1 and self.margin_axis == 2 and self.index == 1)
                     or (y == 1 and self.margin_axis == 1 and self.index == 1)
                     or (z == 1 and self.margin_axis == 0 and self.index == 1)):

            data = self.root_data[z_start: z_end, y_start: y_end, x_start: x_end, :]
            margin_data = np.zeros(1)
            if x == 1 and self.margin_axis == 2:
                margin_data = self.margin[z_start: z_end, y_start: y_end, :, :]
            elif y == 1 and self.margin_axis == 1:
                margin_data = self.margin[z_start: z_end, :, x_start: x_end, :]
            elif z == 1 and self.margin_axis == 0:
                margin_data = self.margin[:, y_start: y_end, x_start: x_end, :]
            data = np.concatenate([margin_data, data], axis=self.margin_axis)
        else:
            data = self.root_data[z_start: z_end, y_start: y_end, x_start: x_end, :]
        return data

    # 나눠진 크기 별로 데이터 및 키 값 리턴
    # TempoGAN 모델에 맞추기 위한 단계
    def set_data_Octree(self):
        key0, data0, key1, data1, key2, data2, key3, data3, key4, data4, key_t, data_t = self._set_data_Octree(self.root)
        return key0, data0, key1, data1, key2, data2, key3, data3, key4, data4, key_t, data_t

    # Tree 순회하며 크기 별로 데이터 모으기
    def _set_data_Octree(self, node):
        depth_tmp = node.key[0]
        depth_pow = pow(2, depth_tmp)
        node_side_size = self.width // depth_pow
        if depth_tmp != self.terminal_depth:

            if node.State == node.Oc_1.State == node.Oc_2.State == node.Oc_3.State == node.Oc_4.State \
                    == node.Oc_5.State == node.Oc_6.State == node.Oc_7.State == node.Oc_8.State == self.S_SRR \
                    and node_side_size <= self.max_patch_size:
                ''''''
                if node.key[0] == 0:
                    self.depth0_SRR_key.append(node.key)
                    self.depth0_SRR_data.append(self.node_data_setting_overlap(node.key))
                elif node.key[0] == 1:
                    self.depth1_SRR_key.append(node.key)
                    self.depth1_SRR_data.append(self.node_data_setting_overlap(node.key))
                elif node.key[0] == 2:
                    self.depth2_SRR_key.append(node.key)
                    self.depth2_SRR_data.append(self.node_data_setting_overlap(node.key))
                elif node.key[0] == 3:
                    self.depth3_SRR_key.append(node.key)
                    self.depth3_SRR_data.append(self.node_data_setting_overlap(node.key))
                elif node.key[0] == 4:
                    self.depth4_SRR_key.append(node.key)
                    self.depth4_SRR_data.append(self.node_data_setting_overlap(node.key))
            elif node.State == self.S_Empty:
                ''''''
            else:
                self._set_data_Octree(node.Oc_1)
                self._set_data_Octree(node.Oc_2)
                self._set_data_Octree(node.Oc_3)
                self._set_data_Octree(node.Oc_4)
                self._set_data_Octree(node.Oc_5)
                self._set_data_Octree(node.Oc_6)
                self._set_data_Octree(node.Oc_7)
                self._set_data_Octree(node.Oc_8)
        else:
            if node.State == self.S_Empty:
                ''''''
            else:
                self.terminal_SRR_key.append(node.key)
                self.terminal_SRR_data.append(self.node_data_setting_overlap(node.key))
        return self.depth0_SRR_key, self.depth0_SRR_data, self.depth1_SRR_key, self.depth1_SRR_data,\
               self.depth2_SRR_key, self.depth2_SRR_data, self.depth3_SRR_key, self.depth3_SRR_data,\
               self.depth4_SRR_key, self.depth4_SRR_data, self.terminal_SRR_key, self.terminal_SRR_data

    # Overlap 부분 제거하면서 Data 합치기
    def data_sum_overlap(self, key, bool_key_list, data, width, height, depth):
        overlap_size = self.overlap_size
        base = np.zeros((width, height, depth, 1), self.data_type)
        array_size = len(key)
        if bool_key_list == 1:
            for i in range(array_size):
                side_size = width // pow(2, key[i][0])
                x = key[i][1]
                y = key[i][2]
                z = key[i][3]
                data_ = data[i]

                base_x_start = (x - 1) * side_size
                base_y_start = (y - 1) * side_size
                base_z_start = (z - 1) * side_size

                base_x_end = x * side_size
                base_y_end = y * side_size
                base_z_end = z * side_size

                data_x_start = overlap_size * 4
                data_y_start = overlap_size * 4
                data_z_start = overlap_size * 4

                data_x_end = overlap_size * 4 + side_size
                data_y_end = overlap_size * 4 + side_size
                data_z_end = overlap_size * 4 + side_size

                # 각각 데이터가 Boundary에 있을 때 따로 설정
                if x == 1:
                    base_x_start = 0
                    base_x_end = side_size
                    if len(self.margin) > 0 and self.margin_axis == 2 and self.index == 1:
                        data_x_start = overlap_size * 4
                        data_x_end = side_size + overlap_size * 4
                    else:
                        data_x_start = 0
                        data_x_end = side_size
                if y == 1:
                    base_y_start = 0
                    base_y_end = side_size
                    if len(self.margin) > 0 and self.margin_axis == 1 and self.index == 1:
                        data_y_start = overlap_size * 4
                        data_y_end = side_size + overlap_size * 4
                    else:
                        data_y_start = 0
                        data_y_end = side_size
                if z == 1:
                    base_z_start = 0
                    base_z_end = side_size
                    if len(self.margin) > 0 and self.margin_axis == 0 and self.index == 1:
                        data_z_start = overlap_size * 4
                        data_z_end = side_size + overlap_size * 4
                    else:
                        data_z_start = 0
                        data_z_end = side_size

                if x * side_size + overlap_size > self.width * 4:
                    base_x_start = (x - 1) * side_size
                    base_x_end = self.width * 4
                    if len(self.margin) > 0 and self.margin_axis == 2 and self.index == 0:
                        data_x_start = overlap_size * 4
                        data_x_end = side_size + overlap_size * 4
                    else:
                        data_x_start = overlap_size * 4 * 2
                        data_x_end = side_size + overlap_size * 4 * 2
                if y * side_size + overlap_size > self.width * 4:
                    base_y_start = (y - 1) * side_size
                    base_y_end = self.width * 4
                    if len(self.margin) > 0 and self.margin_axis == 1 and self.index == 0:
                        data_y_start = overlap_size * 4
                        data_y_end = side_size + overlap_size * 4
                    else:
                        data_y_start = overlap_size * 4 * 2
                        data_y_end = side_size + overlap_size * 4 * 2
                if z * side_size + overlap_size > self.width * 4:
                    base_z_start = (z - 1) * side_size
                    base_z_end = self.width * 4
                    if len(self.margin) > 0 and self.margin_axis == 0 and self.index == 0:
                        data_z_start = overlap_size * 4
                        data_z_end = side_size + overlap_size * 4
                    else:
                        data_z_start = overlap_size * 4 * 2
                        data_z_end = side_size + overlap_size * 4 * 2
                base[base_z_start: base_z_end, base_y_start: base_y_end, base_x_start: base_x_end, :]\
                    = data_[data_z_start: data_z_end, data_y_start: data_y_end, data_x_start: data_x_end, :]
        else:
            side_size = width // pow(2, key[0])
            x = key[1]
            y = key[2]
            z = key[3]
            data_ = data

            x_start = (x - 1) * side_size - overlap_size * 4
            y_start = (y - 1) * side_size - overlap_size * 4
            z_start = (z - 1) * side_size - overlap_size * 4

            x_end = x * side_size + overlap_size * 4
            y_end = y * side_size + overlap_size * 4
            z_end = z * side_size + overlap_size * 4

            if x == 1:
                x_start = (x - 1) * side_size
            if y == 1:
                y_start = (y - 1) * side_size
            if z == 1:
                z_start = (z - 1) * side_size

            if x * side_size + overlap_size > self.width:
                x_end = x * side_size
            if y * side_size + overlap_size > self.width:
                y_end = y * side_size
            if z * side_size + overlap_size > self.width:
                z_end = z * side_size

            base[z_start: z_end, y_start: y_end, x_start: x_end, :] = data_

        return base

    def run(self):
        self.Set_terminal_node_GPU()
        self.Set_tree(self.root)
