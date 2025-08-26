import carla

client = carla.Client('localhost', 2000)

print(client.show_recorder_file_info("/home/ads/ads_testing/log_files/ARG_Carcarana-1_1_I-1-1.log", False))
