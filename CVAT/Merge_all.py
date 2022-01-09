import json

class Merge_all:

    def __init__(self):

        self.object_id = 0
        self.image_id = 0

        self.final = {"licenses": [
                        {
                            "name": "",
                            "id": 0,
                            "url": ""
                        }
                    ],
                    "info": {
                        "contributor": "",
                        "date_created": "",
                        "description": "",
                        "url": "",
                        "version": "",
                        "year": ""
                    },
                    "categories": [
                        {
                            "id": 1,
                            "name": "spl_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 2,
                            "name": "spl_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 3,
                            "name": "spl_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 4,
                            "name": "spl_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 5,
                            "name": "spl_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 6,
                            "name": "spl_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 7,
                            "name": "spl_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 8,
                            "name": "spl_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 9,
                            "name": "spl_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 10,
                            "name": "spl_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 11,
                            "name": "spl_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 12,
                            "name": "spl_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 13,
                            "name": "spl_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 14,
                            "name": "spl_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 15,
                            "name": "spl_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 16,
                            "name": "spl_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 17,
                            "name": "spl_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 18,
                            "name": "spl_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 19,
                            "name": "spl_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 20,
                            "name": "spl_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 21,
                            "name": "spl_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 22,
                            "name": "spl_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 23,
                            "name": "spl_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 24,
                            "name": "spl_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 25,
                            "name": "spl_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 26,
                            "name": "spl_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 27,
                            "name": "spl_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 28,
                            "name": "spl_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 29,
                            "name": "spl_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 30,
                            "name": "spl_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 31,
                            "name": "spl_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 32,
                            "name": "spl_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 33,
                            "name": "spl_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 34,
                            "name": "spl_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 35,
                            "name": "spl_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 36,
                            "name": "spl_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 37,
                            "name": "spl_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 38,
                            "name": "spl_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 39,
                            "name": "spl_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 40,
                            "name": "spl_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 41,
                            "name": "spl_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 42,
                            "name": "spl_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 43,
                            "name": "spl_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 44,
                            "name": "spl_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 45,
                            "name": "spl_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 46,
                            "name": "spl_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 47,
                            "name": "spl_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 48,
                            "name": "spl_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 49,
                            "name": "spl_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 50,
                            "name": "spl_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 51,
                            "name": "spl_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 52,
                            "name": "spl_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 53,
                            "name": "spl_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 54,
                            "name": "spl_gray_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 55,
                            "name": "cone_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 56,
                            "name": "cone_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 57,
                            "name": "cone_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 58,
                            "name": "cone_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 59,
                            "name": "cone_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 60,
                            "name": "cone_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 61,
                            "name": "cone_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 62,
                            "name": "cone_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 63,
                            "name": "cone_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 64,
                            "name": "cone_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 65,
                            "name": "cone_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 66,
                            "name": "cone_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 67,
                            "name": "cone_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 68,
                            "name": "cone_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 69,
                            "name": "cone_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 70,
                            "name": "cone_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 71,
                            "name": "cone_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 72,
                            "name": "cone_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 73,
                            "name": "cone_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 74,
                            "name": "cone_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 75,
                            "name": "cone_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 76,
                            "name": "cone_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 77,
                            "name": "cone_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 78,
                            "name": "cone_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 79,
                            "name": "cone_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 80,
                            "name": "cone_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 81,
                            "name": "cone_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 82,
                            "name": "cone_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 83,
                            "name": "cone_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 84,
                            "name": "cone_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 85,
                            "name": "cone_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 86,
                            "name": "cone_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 87,
                            "name": "cone_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 88,
                            "name": "cone_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 89,
                            "name": "cone_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 90,
                            "name": "cone_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 91,
                            "name": "cone_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 92,
                            "name": "cone_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 93,
                            "name": "cone_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 94,
                            "name": "cone_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 95,
                            "name": "cone_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 96,
                            "name": "cone_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 97,
                            "name": "cone_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 98,
                            "name": "cone_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 99,
                            "name": "cone_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 100,
                            "name": "cone_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 101,
                            "name": "cone_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 102,
                            "name": "cone_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 103,
                            "name": "cone_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 104,
                            "name": "cone_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 105,
                            "name": "cone_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 106,
                            "name": "cone_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 107,
                            "name": "cone_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 108,
                            "name": "cone_gray_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 109,
                            "name": "cube_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 110,
                            "name": "cube_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 111,
                            "name": "cube_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 112,
                            "name": "cube_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 113,
                            "name": "cube_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 114,
                            "name": "cube_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 115,
                            "name": "cube_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 116,
                            "name": "cube_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 117,
                            "name": "cube_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 118,
                            "name": "cube_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 119,
                            "name": "cube_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 120,
                            "name": "cube_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 121,
                            "name": "cube_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 122,
                            "name": "cube_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 123,
                            "name": "cube_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 124,
                            "name": "cube_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 125,
                            "name": "cube_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 126,
                            "name": "cube_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 127,
                            "name": "cube_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 128,
                            "name": "cube_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 129,
                            "name": "cube_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 130,
                            "name": "cube_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 131,
                            "name": "cube_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 132,
                            "name": "cube_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 133,
                            "name": "cube_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 134,
                            "name": "cube_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 135,
                            "name": "cube_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 136,
                            "name": "cube_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 137,
                            "name": "cube_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 138,
                            "name": "cube_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 139,
                            "name": "cube_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 140,
                            "name": "cube_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 141,
                            "name": "cube_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 142,
                            "name": "cube_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 143,
                            "name": "cube_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 144,
                            "name": "cube_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 145,
                            "name": "cube_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 146,
                            "name": "cube_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 147,
                            "name": "cube_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 148,
                            "name": "cube_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 149,
                            "name": "cube_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 150,
                            "name": "cube_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 151,
                            "name": "cube_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 152,
                            "name": "cube_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 153,
                            "name": "cube_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 154,
                            "name": "cube_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 155,
                            "name": "cube_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 156,
                            "name": "cube_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 157,
                            "name": "cube_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 158,
                            "name": "cube_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 159,
                            "name": "cube_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 160,
                            "name": "cube_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 161,
                            "name": "cube_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 162,
                            "name": "cube_gray_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 163,
                            "name": "sphere_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 164,
                            "name": "sphere_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 165,
                            "name": "sphere_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 166,
                            "name": "sphere_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 167,
                            "name": "sphere_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 168,
                            "name": "sphere_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 169,
                            "name": "sphere_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 170,
                            "name": "sphere_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 171,
                            "name": "sphere_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 172,
                            "name": "sphere_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 173,
                            "name": "sphere_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 174,
                            "name": "sphere_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 175,
                            "name": "sphere_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 176,
                            "name": "sphere_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 177,
                            "name": "sphere_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 178,
                            "name": "sphere_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 179,
                            "name": "sphere_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 180,
                            "name": "sphere_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 181,
                            "name": "sphere_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 182,
                            "name": "sphere_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 183,
                            "name": "sphere_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 184,
                            "name": "sphere_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 185,
                            "name": "sphere_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 186,
                            "name": "sphere_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 187,
                            "name": "sphere_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 188,
                            "name": "sphere_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 189,
                            "name": "sphere_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 190,
                            "name": "sphere_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 191,
                            "name": "sphere_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 192,
                            "name": "sphere_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 193,
                            "name": "sphere_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 194,
                            "name": "sphere_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 195,
                            "name": "sphere_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 196,
                            "name": "sphere_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 197,
                            "name": "sphere_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 198,
                            "name": "sphere_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 199,
                            "name": "sphere_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 200,
                            "name": "sphere_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 201,
                            "name": "sphere_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 202,
                            "name": "sphere_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 203,
                            "name": "sphere_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 204,
                            "name": "sphere_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 205,
                            "name": "sphere_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 206,
                            "name": "sphere_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 207,
                            "name": "sphere_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 208,
                            "name": "sphere_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 209,
                            "name": "sphere_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 210,
                            "name": "sphere_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 211,
                            "name": "sphere_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 212,
                            "name": "sphere_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 213,
                            "name": "sphere_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 214,
                            "name": "sphere_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 215,
                            "name": "sphere_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 216,
                            "name": "sphere_gray_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 217,
                            "name": "cylinder_gold_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 218,
                            "name": "cylinder_gold_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 219,
                            "name": "cylinder_gold_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 220,
                            "name": "cylinder_gold_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 221,
                            "name": "cylinder_gold_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 222,
                            "name": "cylinder_gold_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 223,
                            "name": "cylinder_blue_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 224,
                            "name": "cylinder_blue_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 225,
                            "name": "cylinder_blue_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 226,
                            "name": "cylinder_blue_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 227,
                            "name": "cylinder_blue_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 228,
                            "name": "cylinder_blue_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 229,
                            "name": "cylinder_yellow_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 230,
                            "name": "cylinder_yellow_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 231,
                            "name": "cylinder_yellow_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 232,
                            "name": "cylinder_yellow_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 233,
                            "name": "cylinder_yellow_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 234,
                            "name": "cylinder_yellow_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 235,
                            "name": "cylinder_red_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 236,
                            "name": "cylinder_red_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 237,
                            "name": "cylinder_red_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 238,
                            "name": "cylinder_red_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 239,
                            "name": "cylinder_red_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 240,
                            "name": "cylinder_red_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 241,
                            "name": "cylinder_brown_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 242,
                            "name": "cylinder_brown_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 243,
                            "name": "cylinder_brown_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 244,
                            "name": "cylinder_brown_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 245,
                            "name": "cylinder_brown_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 246,
                            "name": "cylinder_brown_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 247,
                            "name": "cylinder_green_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 248,
                            "name": "cylinder_green_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 249,
                            "name": "cylinder_green_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 250,
                            "name": "cylinder_green_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 251,
                            "name": "cylinder_green_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 252,
                            "name": "cylinder_green_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 253,
                            "name": "cylinder_cyan_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 254,
                            "name": "cylinder_cyan_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 255,
                            "name": "cylinder_cyan_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 256,
                            "name": "cylinder_cyan_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 257,
                            "name": "cylinder_cyan_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 258,
                            "name": "cylinder_cyan_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 259,
                            "name": "cylinder_purple_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 260,
                            "name": "cylinder_purple_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 261,
                            "name": "cylinder_purple_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 262,
                            "name": "cylinder_purple_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 263,
                            "name": "cylinder_purple_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 264,
                            "name": "cylinder_purple_large_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 265,
                            "name": "cylinder_gray_small_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 266,
                            "name": "cylinder_gray_small_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 267,
                            "name": "cylinder_gray_medium_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 268,
                            "name": "cylinder_gray_medium_metal",
                            "supercategory": ""
                        },
                        {
                            "id": 269,
                            "name": "cylinder_gray_large_rubber",
                            "supercategory": ""
                        },
                        {
                            "id": 270,
                            "name": "cylinder_gray_large_metal",
                            "supercategory": ""
                        }
                    ],
                    'images': [],
                    'annotations': []}
    
    def open_file(self, input_path: str):

        with open(input_path, 'r', encoding = 'UTF-8') as cvat_file:
            piece = json.load(cvat_file)
            cvat_file.close()
        
        return piece

    def add_images_and_annotations(self, input_path: str):
        
        piece = self.open_file(input_path)

        for image in piece['images']:
            
            self.image_id = self.image_id + 1
            
            for annotation in piece['annotations']:
                if annotation['image_id'] == image['id']:
                    self.object_id = self.object_id+1
                    annotation['id'] = self.object_id
                    annotation['image_id'] = self.image_id
                    self.final['annotations'].append(annotation)   
                else:
                    pass

            image['id'] = self.image_id
            self.final['images'].append(image)
    
    def save(self, output_path: str):
       
        output = json.dumps(self.final)
        output_file = open(output_path, 'w', encoding = 'UTF-8')
        output_file.write(output)
        output_file.close()

    def merge_consist(self, start: int, end: int):
        
        output_path = './{}-{}.json'.format(start,end)
        for i in range(start,int(end-3),5):
            input_path = './d_data/{}-{}.json'.format(str(i),str(int(i+4)))
            self.add_images_and_annotations(input_path)
            self.save(output_path)
            print(2*'>>>>>>>>>'+'merge finish')
            print('success!!!')

    def merge_arbitrary(self, file1_start: int, file1_end: int, file2_start: int, file2_end: int):
        
        input_path1 = './d_data/{}-{}.json'.format(str(file1_start),str(file1_end))
        input_path2 = './d_data/{}-{}.json'.format(str(file2_start),str(file2_end))
        output_path = './{}-{}.json'.format(file1_start,file2_end)
        
        self.add_images_and_annotations(input_path1)
        self.add_images_and_annotations(input_path2)
        self.save(output_path)
        print(2*'>>>>>>>>>'+'merge finish')
        print('success!!!')

    
if __name__ == "__main__":
    
    ma = Merge_all()
    
    '''
    # always 5 files as a unit
    # if start with '5200-5204.json' and end with '5205-5209.json'
    ma.merge_consist(start=5200, end=5224)

    '''
    # if some files are missed before, then use this code to merge them
    # e.g. if file1 = '5215-5240.json'
    ma.merge_arbitrary(file1_start=5200, file1_end=5204, file2_start= 5215, file2_end= 5219)