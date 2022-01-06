# CATER Object detection 

use the following command to connect to server, password required (provided in the email) if not using SSH key

sh ./connect_server.sh


### directory_format:

```
.
├── *detectron2 -> /<Your Dir For Detectron2>/ 
├── jupyter-notebook
├── *output
├── *dataset
│   ├── annotations
│   └── images
│    └── image 
├── *raw_data
│   ├── raw_data_from_005200_to_005699
│   │   ├── 005200-005299
│   │   ├── 005300-005399
│   │   ├── 005400-005499
│   │   ├── 005500-005599
│   │   └── 005600-005699
│   ├── all_action_camera_move
│   │   ├── lists
│   │   │   └── localize
│   │   ├── scenes
│   │   └── videos
│   ├── cocotest
│   │   ├── annotations
│   │   └── images
│   │       └── image
│   ├── first_frame
│   │   └── all_actions_first_frame
│   └── mydata
├── scripts
└── test
    └── __pycache__
*: must have

```