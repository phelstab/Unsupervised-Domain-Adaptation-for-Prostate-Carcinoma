UDA shell scripts for new split:
  source = RUMC+PCNN+ZGT
  target = UULM

Default entry point:
  scripts/runners/uda_sh_new/main.sh

Individual algorithm runners:
- 2_mcd.sh
- 3_dann.sh
- 4_mmd.sh
- 5_hybrid.sh
- 6_mcc.sh
- 7_bnm.sh
- 8_daarda.sh

Requirements:
- .venv-cnn prepared
- UULM metadata present at 0ii/data.xlsx
- UULM MRI data present at 0ii/files/mri_data

Outputs:
- Registered tensors for UULM are created in 0ii/files/registered
- Training runs are written under workdir/uda
