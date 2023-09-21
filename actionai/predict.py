import os
import glob
import json
import subprocess
from pkg_resources import resource_filename


def predict(model_dir, window_size, video, save, display):
    actionai_module = "run_actionai.py"
    json_kwarg = json.dumps(
        {"model_dir": model_dir, "window_size": window_size}
    )
    gst_command = f"""gst-launch-1.0 filesrc location={video} ! decodebin ! gvaclassify \
            model=/home/dlstreamer/intel/dl_streamer/models/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml \
            model-proc=/opt/intel/dlstreamer/samples/gstreamer/model_proc/intel/human-pose-estimation-0001.json \
            device=CPU inference-region=full-frame ! queue ! gvapython module={actionai_module} \
            class=ActionAI function=add_pose kwarg='{json_kwarg}' ! queue ! gvametaconvert json-indent=4 \
            ! gvametapublish method=file ! fakesink sync=false"""
    subprocess.run(gst_command, shell=True)

    return f"Ran model saved successfully: {model_dir}/model.h5"
