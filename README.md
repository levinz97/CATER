# CATER Object detection 

use the following command to connect to server, password required (provided in the email) if not using SSH key

sh ./connect_server.sh


### directory_format:

```
.
├── *detectron2 -> /home/levinz/Code/detectron2/ 
├── jupyter-notebook
├── *output
├── *dataset
│   ├── annotations
│   └── images
│    └── image 
├── *raw_data
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
│   │   └── all_actions_first_frame -> /home/levinz/Downloads/all_actions_first_frame/
│   └── mydata
├── scripts
└── test
    └── __pycache__
*: must have

```