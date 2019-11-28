 result 폴더
  Super-resolution을 진행하고 난 결과가 저장됩니다.
  결과 폴더를 GRIDY_VIEWER_infi에 끌어다 넣으면 결과를 확인할 수 있습니다.
  time_check 폴더에는 frame당 소요된 시간이 저장되어 있습니다.

 tools 폴더
  여기에는 TempoGAN Model에 관련된 코드가 들어있습니다. 

사용법)

크기 제한으로 인해 Data와 Model을 올려두지 못했습니다. 원하시는 분은 메일을 주시면 보내드리도록 하겠습니다.
bshong2850@naver.com

저에게 Data와 model 폴더를 메일로 받으신 뒤
main.py를 실행하셔서 

기본 Parameter Setting 원하는대로 하신 뒤
Set.Set_Octree_test_data_GPU_nocube() 함수에 분할하고 싶은 Patch size를 입력하면 됩니다.

보내드릴 파일
 Data 폴더
  미리 만들어놓은 64 x 128 x 64 x 4(density, u, v, w) 크기의 데이터를 만들어 놓았습니다.
  데이터는 Stable Fluids를 활용하여 만든 데이터 입니다.

 model 폴더
  TempoGAN에서 제공하는 Model이 들어있습니다.
  https://github.com/thunil/tempoGAN 여기서 확인할 수 있습니다.