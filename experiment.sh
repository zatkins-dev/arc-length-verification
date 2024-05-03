BASE_DIR=$(dirname "$(readlink -f "$0")")
OUTPUT_DIR="$BASE_DIR/output"
EXE="$BASE_DIR/bin/arch-buckling"
AL_ARGS="-control_type arclength -snes_type newtonal -snes_newtonal_step_size 10"
MONITOR_OPTIONS_AL="-monitor_solution vtk:solution.vts -monitor_force vtk:force.vts -monitor_force_displacement ascii:force_displacement.csv"
MONITOR_OPTIONS_LS="-ts_monitor_solution vtk:solution.vts -monitor_force vtk:force.vts -monitor_force_displacement ascii:force_displacement.csv"

outfile()
{
  echo "arch_${SUFFIX}.out.txt"
}

postrun()
{
  echo "  $(grep 'Nonlinear solve' "$(outfile)")"
  mkdir -p "$OUTPUT_DIR/$SUFFIX"
  mv solution_*.vts "$OUTPUT_DIR/$SUFFIX/"
  mv force_*.vts "$OUTPUT_DIR/$SUFFIX/"
  mv force_displacement.csv "$OUTPUT_DIR/$SUFFIX/"
  mv "$(outfile)" "$OUTPUT_DIR/$SUFFIX/"
}

OLD=$(pwd)
cd "$BASE_DIR" || exit
make -j
cd "$OLD" || exit

rm -rf "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR"
rm -- *.vts *.csv *.out.txt

for load in "-1.5e3" "-2e3"; do
  echo "SNESNEWTONLS, LU"
  SUFFIX="newton_lu_${load}"
  "$EXE" -options_file common.yml -ts_dt 0.01 -ksp_type preonly -pc_type lu -control_type load -ploading $load $MONITOR_OPTIONS_LS > "$(outfile)"
  postrun

  echo "SNESNEWTONLS"
  SUFFIX="newton_${load}"
  "$EXE" -ts_dt 0.01 -options_file common.yml -control_type load -ploading $load $MONITOR_OPTIONS_LS > "$(outfile)"
  postrun

  echo "SNESNEWTONAL exact corrections"
  SUFFIX="arclength_exact_${load}"
  "$EXE" -options_file common.yml ${AL_ARGS} -snes_newtonal_correction_type exact -ploading $load $MONITOR_OPTIONS_AL > "$(outfile)"
  postrun

  echo "SNESNEWTONAL normal corrections"
  SUFFIX="arclength_normal_${load}"
  "$EXE" -options_file common.yml ${AL_ARGS} -snes_newtonal_correction_type normal -ploading $load $MONITOR_OPTIONS_AL > "$(outfile)"
  postrun
done
