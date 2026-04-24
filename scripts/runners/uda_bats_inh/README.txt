UDA batch scripts for new split:
  source = RUMC+PCNN+ZGT
  target = UULM

Default entry point:
  scripts\runners\uda_bats_new\main.bat

Individual algorithm runners:
- 2_mcd.bat
- 3_dann.bat
- 4_mmd.bat
- 5_hybrid.bat
- 6_mcc.bat
- 7_bnm.bat
- 8_daarda.bat

Requirements:
- .venv-cnn prepared
- UULM metadata present at 0ii\data.xlsx
- UULM MRI data present at 0ii\files\mri_data

Outputs:
- Registered tensors for UULM are created in 0ii\files\registered
- Training runs are written under workdir\uda
