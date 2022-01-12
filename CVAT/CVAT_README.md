|-- CVAT
    |-- .wrong.json
    |-- catagories_label.txt
    |-- json_check.py
    |-- cvat_merge.py
    |-- merge_all.py
    
    *|-- json_data
    *|   |-- CATER_new_005200.json
    *|   |-- 放置包含所有视频信息的json文件
    
    *|-- opencv_data
    *|   |-- CATER_new_005200.json
    *|   |-- 初始要放置 generate_data生成的json文件，而后每次运行都会刷新这个文件夹里的文件，始终保持其中文件的最新
    *|   |-- 即：原a_data b_data d_data的合而为一
    
    *|-- cvat_data
    *|   |-- 5200
         |-- anotations
            |-- instances_default.json
    *|   |-- 放置 CVAT导出的json文件
