import numpy as np
import cv2
import os
import timeit
import sys
import tensorflow as tf
from tqdm import tqdm
from pycuda import driver as drv
from pycuda import compiler, gpuarray, tools
import pycuda.autoinit

# load manta tools
sys.path.append("./tools")
import paramhelpers as ph
from GAN_noprint import GAN, lrelu

# 직접 구현한 Octree tools
from Octree_for_SR import octree_for_SR

class super_resolution(object):
    def __init__(self,
                 Set_data_dir="D:\\",
                 full_side_size=64,
                 full_side_size_x=128,
                 full_side_size_y=64,
                 full_side_size_z=64,
                 start_frame=0,
                 total_frame=200,
                 scene="basic3",
                 data_type=np.float32):

        self.Set_data_dir = Set_data_dir
        self.simSizeLow = full_side_size
        self.simSizeLow_x = full_side_size_x
        self.simSizeLow_y = full_side_size_y
        self.simSizeLow_z = full_side_size_z
        self.start_frame = start_frame
        self.total_frame = total_frame
        self.depth = 4
        self.scene = scene

        self.data_type = data_type

        ## Set Octree test data GPU
        self.gpu_num = 0

        self.train = False
        self.dropoutOutput = 1.0
        self.batch_norm = True

        self.upRes = 4  # fixed for now...
        self.simSizeHigh = self.simSizeLow * self.upRes
        self.simSizeHigh_x = self.simSizeLow_x * self.upRes
        self.simSizeHigh_y = self.simSizeLow_y * self.upRes
        self.simSizeHigh_z = self.simSizeLow_z * self.upRes
        self.bn = self.batch_norm
        self.rbId = 0
        self.overlap = 3
        self.set_parameter(22)

    # 폴더 생성 함수
    def make_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # Data Load
    def read_file_nocube(self, data_dir_):
        return np.reshape(np.fromfile(data_dir_, dtype=self.data_type),
                          (self.simSizeLow_z, self.simSizeLow_y, self.simSizeLow_x, self.depth))

    # tileSize에 맞게 Parameter 조정
    def set_parameter(self, tileSizeLow):
        self.tileSizeLow = tileSizeLow
        self.tileSizeHigh = self.tileSizeLow * self.upRes

        self.n_input = self.tileSizeLow ** 3
        self.n_output = self.tileSizeHigh ** 3
        self.n_inputChannels = 4
        self.n_input *= self.n_inputChannels

    # TempoGAN에서 제공하는 generator에 필요한 resBlock
    def resBlock(self, gan, inp, s1, s2, reuse, use_batch_norm, filter_size=3):
        global rbId

        # convolutions of resnet block
        filter = [filter_size, filter_size, filter_size]
        filter1 = [1, 1, 1]

        gc1, _ = gan.convolutional_layer(s1, filter, tf.nn.relu, stride=[1], name="g_cA%d" % rbId, in_layer=inp,
                                         reuse=reuse, batch_norm=use_batch_norm, train=self.train)  # ->16,64
        gc2, _ = gan.convolutional_layer(s2, filter, None, stride=[1], name="g_cB%d" % rbId, reuse=reuse,
                                         batch_norm=use_batch_norm, train=self.train)  # ->8,128

        # shortcut connection
        gs1, _ = gan.convolutional_layer(s2, filter1, None, stride=[1], name="g_s%d" % rbId, in_layer=inp,
                                         reuse=reuse,
                                         batch_norm=use_batch_norm, train=self.train)  # ->16,64
        resUnit1 = tf.nn.relu(tf.add(gc2, gs1))
        rbId += 1
        return resUnit1

    # TempoGAN에서 제공하는 generator Layer
    def gen_resnet(self, _in, num, reuse=False, use_batch_norm=False, train=None):
        global rbId
        rbId = 0
        with tf.variable_scope("generator_" + str(num), reuse=reuse) as scope:
            _in = tf.reshape(_in, shape=[-1, self.tileSizeLow, self.tileSizeLow, self.tileSizeLow, self.n_inputChannels])  # NDHWC
            gan = GAN(_in)

            gan.max_depool()
            inp = gan.max_depool()
            ru1 = self.resBlock(gan, inp, self.n_inputChannels * 2, self.n_inputChannels * 8, reuse, use_batch_norm, 5)

            ru2 = self.resBlock(gan, ru1, 128, 128, reuse, use_batch_norm, 5)
            inRu3 = ru2
            ru3 = self.resBlock(gan, inRu3, 32, 8, reuse, use_batch_norm, 5)
            ru4 = self.resBlock(gan, ru3, 2, 1, reuse, False, 5)
            resF = tf.reshape(ru4, shape=[-1, self.n_output])
            return resF


    # TempoGAN Super-resolution 함수
    def tempoGAN_nocube(self, data__, key__):
        resultTiles = []
        sampler = self.sampler_0
        if self.tileSizeLow == self.patch_size * 1 + self.overlap * 2:
            sampler = self.sampler_0
        elif self.tileSizeLow == self.patch_size * 2 + self.overlap * 2:
            sampler = self.sampler_1
        elif self.tileSizeLow == self.patch_size * 4 + self.overlap * 2:
            sampler = self.sampler_2
        elif self.tileSizeLow == self.patch_size * 8 + self.overlap * 2:
            sampler = self.sampler_3
        elif self.tileSizeLow == self.patch_size * 16 + self.overlap * 2:
            sampler = self.sampler_4

        # Super_resolution 진행
        for tileno in tqdm(range(data__.shape[0])):
            batch_xs_in = np.reshape(data__[tileno], [-1, self.n_input])
            results = self.sess.run(sampler, feed_dict={self.x: batch_xs_in, self.keep_prob: 1.0, self.train: False})
            results = np.array(results)
            results_reshape = np.reshape(results, [self.tileSizeHigh, self.tileSizeHigh, self.tileSizeHigh, 1])

            resultTiles.append(results_reshape)

        # Super_resolution 된 데이터 하나로 합치기
        result_hd = self.Oc_sr.data_sum_overlap(key__, 1, resultTiles, self.simSizeHigh, self.simSizeHigh, self.simSizeHigh)

        return result_hd


    # 패치 별 최댓값 구하는 함수
    # GPU Reduction을 활용하여 최댓값 구하기
    def Gen_Maxmatrix_nocube(self, patch_size):
        # Pycuda 코드
        func = """
                    #include <stdio.h>
                    __global__ void reduction_max(float *input,int input_N, float *output)
                    {
                        unsigned int size = input_N;
                        __shared__ float Max_x[4][4][4];
                        unsigned int output_size = size * 0.5;

                        unsigned int tz = threadIdx.z + blockDim.z * blockIdx.z;
                        unsigned int ty = threadIdx.y + blockDim.y * blockIdx.y;
                        unsigned int tx = threadIdx.x + blockDim.x * blockIdx.x;

                        unsigned int tid_x = threadIdx.x;
                        unsigned int tid_y = threadIdx.y;
                        unsigned int tid_z = threadIdx.z;
                        Max_x[tid_x][tid_y][tid_z] = input[tx + (ty * size) + (tz * size * size)];
                        __syncthreads();

                        for(unsigned int s=1; s < blockDim.x; s*=2)
                        {   
                            int index = 2 * s * tid_x;
                            if(index < blockDim.x)
                            {
                                Max_x[tid_x][tid_y][tid_z] = (Max_x[tid_x + s][tid_y][tid_z] > Max_x[tid_x][tid_y][tid_z])
                                 ? Max_x[tid_x + s][tid_y][tid_z] : Max_x[tid_x][tid_y][tid_z];
                            }
                        }

                        if(tid_x == 0)
                        {
                            input[tx + (ty * size) + (tz * size * size)] = Max_x[tid_x][tid_y][tid_z];
                        }

                        Max_x[tid_x][tid_y][tid_z] = input[tx + (ty * size) + (tz * size * size)];
                        __syncthreads();        

                        for(unsigned int r=1; r < blockDim.y; r*=2)
                        {      
                            int index = 2 * r * tid_y;
                            if(index < blockDim.y)
                            {
                                Max_x[tid_x][tid_y][tid_z] = (Max_x[tid_x][tid_y + r][tid_z] > Max_x[tid_x][tid_y][tid_z])
                                 ? Max_x[tid_x][tid_y + r][tid_z] : Max_x[tid_x][tid_y][tid_z];
                            }
                        }

                        if(tid_x == 0)
                        {
                            if(tid_y == 0)
                            {
                                input[tx + (ty * size) + (tz * size * size)] = Max_x[tid_x][tid_y][tid_z];
                            }
                        }

                        Max_x[tid_x][tid_y][tid_z] = input[tx + (ty * size) + (tz * size * size)];
                        __syncthreads();      

                        for(unsigned int r=1; r < blockDim.z; r*=2)
                        {      
                            int index = 2 * r * tid_z;
                            if(index < blockDim.z)
                            {
                                Max_x[tid_x][tid_y][tid_z] = (Max_x[tid_x][tid_y][tid_z + r] > Max_x[tid_x][tid_y][tid_z])
                                 ? Max_x[tid_x][tid_y][tid_z + r] : Max_x[tid_x][tid_y][tid_z];
                            }
                        }

                        if(tid_x == 0)
                        {
                            if(tid_y == 0)
                            {
                                if(tid_z == 0)
                                {
                                    output[blockIdx.x + blockIdx.y * output_size + blockIdx.z * output_size * output_size]
                                     = Max_x[tid_x][tid_y][tid_z];
                                    __syncthreads();
                                }
                            }
                        }

                    }

                    """

        if (os.system("cl.exe")):
            os.environ[
                'PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\HostX64\x64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")

        mod = compiler.SourceModule(func)
        reduction_max = mod.get_function("reduction_max")

        data_shape_size = self.simSizeLow // patch_size

        # 원하는 패치 개수만큼 최댓값 굳하는 함수
        def Max(input_data, input_N):
            input = input_data
            block_N = 2
            grid_N = int(input_N / block_N)
            output = np.zeros((input_N // 2, input_N // 2, input_N // 2), np.float32)
            reduction_max(drv.In(input), np.int32(input_N), drv.Out(output), block=(block_N, block_N, block_N),
                          grid=(grid_N, grid_N, grid_N))
            if (output.shape[0] == data_shape_size):
                return output

            input = output
            input_N = input.shape[0]
            result = Max(input, input_N)

            return result

        # 원하는 프레임만큼 돌면서 Max_Martix를 만들어서 리스트로 저장
        Max_Matrix_list1 = []
        Max_Matrix_list2 = []
        for frame in range(self.start_frame, self.total_frame):
            data_dir_ = self.Set_data_dir + "%05d.dv" % (frame + 1)
            data = self.read_file_nocube(data_dir_)
            data1 = np.zeros(1)
            data2 = np.zeros(1)
            if self.simSizeLow_x > self.simSizeLow_y and self.simSizeLow_x > self.simSizeLow_z:
                data1 = np.reshape(data[:, :, 0:self.simSizeLow_x // 2, 0], (self.simSizeLow_z, self.simSizeLow_y, self.simSizeLow_x // 2, 1))
                data2 = np.reshape(data[:, :, self.simSizeLow_x // 2: , 0], (self.simSizeLow_z, self.simSizeLow_y, self.simSizeLow_x // 2, 1))
            elif self.simSizeLow_y > self.simSizeLow_x and self.simSizeLow_y > self.simSizeLow_z:
                data1 = np.reshape(data[:, 0:self.simSizeLow_y // 2, :, 0], (self.simSizeLow_z, self.simSizeLow_y // 2, self.simSizeLow_x, 1))
                data2 = np.reshape(data[:, self.simSizeLow_y // 2: , :, 0], (self.simSizeLow_z, self.simSizeLow_y // 2, self.simSizeLow_x, 1))
            elif self.simSizeLow_z > self.simSizeLow_x and self.simSizeLow_z > self.simSizeLow_y:
                data1 = np.reshape(data[0:self.simSizeLow_z // 2, :, :, 0], (self.simSizeLow_z // 2, self.simSizeLow_y, self.simSizeLow_x, 1))
                data2 = np.reshape(data[self.simSizeLow_z // 2: , :, :, 0], (self.simSizeLow_z // 2, self.simSizeLow_y, self.simSizeLow_x, 1))
            data1 = np.ascontiguousarray(data1, dtype=np.float32)
            data2 = np.ascontiguousarray(data2, dtype=np.float32)

            Max_Matrix1 = Max(data1, self.simSizeLow)
            Max_Matrix_list1.append(Max_Matrix1)
            Max_Matrix2 = Max(data2, self.simSizeLow)
            Max_Matrix_list2.append(Max_Matrix2)
        return Max_Matrix_list1, Max_Matrix_list2


    # ocTree를 활용한 Super-resolution 함수
    def Set_Octree_test_data_GPU_nocube(self, patch_size):
        self.patch_size = patch_size
        self.save_dir = "result/" + self.scene + "_" + str(patch_size)
        self.make_folder(self.save_dir)
        f_time = open("result/time_check/" + self.scene + "_" + str(patch_size) + ".txt", 'w')
        pre_process_time_start = timeit.default_timer()
        ph.checkUnusedParams()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_num)

        modelPath = "model/model_0034_final.ckpt"
        start_frame = self.start_frame
        end_frame = self.total_frame

        f_h = open(self.save_dir + "/00000.GRIDY_INIT", 'w')
        f_h.write("3 " + str(self.simSizeLow_x * 4 - 2) + " " + str(self.simSizeLow_y * 4 - 2)
                  + " " + str(self.simSizeLow_z * 4 - 2) + "\n")
        f_h.write(str(start_frame + 1) + " " + str(end_frame) + "\n")
        f_h.write("1")
        f_h.close()

        # 각 패치 별 최댓값으로 만든 메트릭스 생성
        Max_Matrix_list1, Max_Matrix_list2 = self.Gen_Maxmatrix_nocube(patch_size)
        Max_Matrix_list_size = 0

        self.x = tf.placeholder(tf.float32, shape=[None, None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.train = tf.placeholder(tf.bool)

        # 각 패치 크기 별로 Layer 설정
        self.set_parameter(self.patch_size * 1 + self.overlap * 2)
        self.sampler_0 = self.gen_resnet(self.x, 0, reuse=False, use_batch_norm=True, train=False)
        self.set_parameter(self.patch_size * 2 + self.overlap * 2)
        self.sampler_1 = self.gen_resnet(self.x, 1, reuse=False, use_batch_norm=True, train=False)
        self.set_parameter(self.patch_size * 4 + self.overlap * 2)
        self.sampler_2 = self.gen_resnet(self.x, 2, reuse=False, use_batch_norm=True, train=False)
        self.set_parameter(self.patch_size * 8 + self.overlap * 2)
        self.sampler_3 = self.gen_resnet(self.x, 3, reuse=False, use_batch_norm=True, train=False)
        self.set_parameter(self.patch_size * 16 + self.overlap * 2)
        self.sampler_4 = self.gen_resnet(self.x, 4, reuse=False, use_batch_norm=True, train=False)

        # TempoGAN 모델 불러오기
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        saver = tf.train.Saver()
        saver.restore(self.sess, modelPath)

        pre_process_time = timeit.default_timer() - pre_process_time_start

        f_time.write("0" + " " + str(pre_process_time))
        for frame in range(self.start_frame, self.total_frame):
            print("frame = ", frame)
            one_frame_start_time = timeit.default_timer()

            data_dir_ = self.Set_data_dir + "%05d.dv" % (frame + 1)
            data = self.read_file_nocube(data_dir_)
            data1 = np.zeros(1)
            data2 = np.zeros(1)
            margin1 = np.zeros(1)
            margin2 = np.zeros(1)
            axis = 2

            # cubic이 아닌 데이터는 자른 위치에 따라 맞게 overlap설정
            if self.simSizeLow_x > self.simSizeLow_y and self.simSizeLow_x > self.simSizeLow_z:
                data1 = np.reshape(data[:, :, 0:self.simSizeLow_x // 2, :], (self.simSizeLow_z, self.simSizeLow_y, self.simSizeLow_x // 2, 4))
                data2 = np.reshape(data[:, :, self.simSizeLow_x // 2: , :], (self.simSizeLow_z, self.simSizeLow_y, self.simSizeLow_x // 2, 4))
                margin1 = np.reshape(data[:, :, self.simSizeLow_x // 2: self.simSizeLow_x // 2 + self.overlap, :], (self.simSizeLow_z, self.simSizeLow_y, self.overlap, 4))
                margin2 = np.reshape(data[:, :, self.simSizeLow_x // 2 - self.overlap: self.simSizeLow_x // 2, :], (self.simSizeLow_z, self.simSizeLow_y, self.overlap, 4))

            elif self.simSizeLow_y > self.simSizeLow_x and self.simSizeLow_y > self.simSizeLow_z:
                data1 = np.reshape(data[:, 0:self.simSizeLow_y // 2, :, :], (self.simSizeLow_z, self.simSizeLow_y // 2, self.simSizeLow_x, 4))
                data2 = np.reshape(data[:, self.simSizeLow_y // 2: , :, :], (self.simSizeLow_z, self.simSizeLow_y // 2, self.simSizeLow_x, 4))
                margin1 = np.reshape(data[:, self.simSizeLow_y // 2: self.simSizeLow_y // 2 + self.overlap, :, :], (self.simSizeLow_z, self.overlap, self.simSizeLow_x, 4))
                margin2 = np.reshape(data[:, self.simSizeLow_y // 2 - self.overlap: self.simSizeLow_y // 2, :, :], (self.simSizeLow_z, self.overlap, self.simSizeLow_x, 4))
                axis = 1
            elif self.simSizeLow_z > self.simSizeLow_x and self.simSizeLow_z > self.simSizeLow_y:
                data1 = np.reshape(data[0:self.simSizeLow_z // 2, :, :, :], (self.simSizeLow_z // 2, self.simSizeLow_y, self.simSizeLow_x, 4))
                data2 = np.reshape(data[self.simSizeLow_z // 2: , :, :, :], (self.simSizeLow_z // 2, self.simSizeLow_y, self.simSizeLow_x, 4))
                margin1 = np.reshape(data[self.simSizeLow_z // 2: self.simSizeLow_z // 2 + self.overlap, :, :, :], (self.overlap, self.simSizeLow_y, self.simSizeLow_x, 4))
                margin2 = np.reshape(data[self.simSizeLow_z // 2 - self.overlap: self.simSizeLow_x // 2, :, :, :], (self.overlap, self.simSizeLow_y, self.simSizeLow_x, 4))
                axis = 0
            data1 = np.ascontiguousarray(data1, dtype=np.float32)
            data2 = np.ascontiguousarray(data2, dtype=np.float32)
            margin1 = np.ascontiguousarray(margin1, dtype=np.float32)
            margin2 = np.ascontiguousarray(margin2, dtype=np.float32)

            def super_resolution(data__, MML, MML_size, margin, axis, index):
                self.Oc_sr =octree_for_SR(data__, self.data_type, MML[MML_size], patch_size, margin, axis, index)

                # 패치 크기 및 키 별로 받기
                key0, data0, key1, data1, key2, data2, key3, data3, key4, data4, key_t, data_t = self.Oc_sr.set_data_Octree()
                data0 = np.array(data0)
                data1 = np.array(data1)
                data2 = np.array(data2)
                data3 = np.array(data3)
                data4 = np.array(data4)
                data_t = np.array(data_t)

                # 결과로 저장할 배열 설정
                final_result = np.zeros([self.simSizeLow * 4, self.simSizeLow * 4, self.simSizeLow * 4, 1]).astype(
                    np.float32)

                # 패치 크기마다 데이터가 있을 때 Super-resolution
                # final result에 각각 더해주기
                if data0.shape[0] > 0:
                    self.set_parameter(data0.shape[1])
                    result0 = self.tempoGAN_nocube(data0, key0)
                    final_result += result0
                if data1.shape[0] > 0:
                    self.set_parameter(data1.shape[1])
                    result1 = self.tempoGAN_nocube(data1, key1)
                    final_result += result1
                if data2.shape[0] > 0:
                    self.set_parameter(data2.shape[1])
                    result2 = self.tempoGAN_nocube(data2, key2)
                    final_result += result2
                if data3.shape[0] > 0:
                    self.set_parameter(data3.shape[1])
                    result3 = self.tempoGAN_nocube(data3, key3)
                    final_result += result3
                if data4.shape[0] > 0:
                    self.set_parameter(data4.shape[1])
                    result4 = self.tempoGAN_nocube(data4, key4)
                    final_result += result4
                if data_t.shape[0] > 0:
                    self.set_parameter(data_t.shape[1])
                    result_t = self.tempoGAN_nocube(data_t, key_t)
                    final_result += result_t

                return final_result

            final_result1 = super_resolution(data1, Max_Matrix_list1, Max_Matrix_list_size, margin1, axis, 0)
            final_result2 = super_resolution(data2, Max_Matrix_list2, Max_Matrix_list_size, margin2, axis, 1)
            final_result_sum = np.concatenate([final_result1, final_result2], axis=axis)

            # 결과 저장
            final_result_sum.tofile(self.save_dir + "/%05d" % (frame + 1) + ".density")
            one_frame_time = timeit.default_timer() - one_frame_start_time
            f_time.write("\n" + str(frame+1) + " " + str(one_frame_time))
            Max_Matrix_list_size += 1

        f_time.close()
        self.sess.close()
