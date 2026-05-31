## Tasks:
- Simulation
* Ball start on kendama (in progress, Rizon4r_with_kendama.urdf) -> throw ball up -> catch ball
* Ball start with initial upward velocity -> catch ball

# Instructions

- `./launch_titania-4s_robot_only.sh` in `~/OpenSai/drivers/FlexivRizonRedisDriver/redis_driver`
- `scripts/launch.sh kendama.xml` in `~/OpenSai` (need to move kendama.xml to `~/OpenSai/config_folder/xml_config_files`
- `~OpenSai/drivers/FlexivRizonRedisDriver/redis_driver/driver_robot_only.cpp` for position, velocity, torque limits etc. 

misc: `gsettings set org.gnome.desktop.input-sources xkb-options "['caps:escape']"` for vi use 
