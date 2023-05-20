import logging as lg
import os
from datetime import datetime

project_name = "deliverytime_ML_model"
modelname = "LinerRegression2"

logfilename = f"{datetime.now().strftime('%m_%d_%Y_')}.log"
path_ = os.getcwd()+"/"+project_name+"/logs/"+modelname



os.makedirs(path_, exist_ok=True)
log_dir = os.path.join(path_,logfilename)


logfilepath = os.path.join(log_dir)


lg.basicConfig(
    filename=logfilepath,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=lg.INFO
)

