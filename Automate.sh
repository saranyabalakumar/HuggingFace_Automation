cd
export HOME_DIR=$(pwd) 
cd $HOME_DIR/ 
git clone https://github.com/saranyabalakumar/HuggingFace_Automation.git 
conda deactivate 
cd $HOME_DIR/HuggingFace_Automation/
python3 -m venv HF  
source $HOME_DIR/HuggingFace_Automation/HF/bin/activate 
pip install -r requirements.txt 
python3 HuggingFace_TextModels.py > $HOME_DIR/HuggingFace_Automation/Log/HuggingFace_TextModels.txt  
cd $HOME_DIR/HuggingFace_Automation/Log/  
python3 HuggingFace_TextModels_CSV.py  
