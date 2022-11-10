#!/usr/bin/env bash
#SBATCH -J RWrangle
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o %j.%N.%a.rainwrangle.out.log
#SBATCH -e %j.%N.%a.rainwrangle.err.log
#SBATCH -p compute
#SBATCH --time=3-00:00:00
#SBATCH --mem=8096
# * 8GB RAM

set -e;

module load utilities/multi
module load readline/7.0
module load gcc/10.2.0

# module load cuda/11.5.0

module load python/anaconda/4.6/miniconda/3.7

RAINFALL="${RAINFALL:-$HOME/data/nimrod_ceda.jsonl.gz}";
WATER="${WATER:-$HOME/data/WaterDepths-new.stream.asc.gz}";
OUTPUT="${OUTPUT}";
COUNT_FILE="${COUNT_FILE:-4096}";

if [[ -z "${WATER}" ]]; then
	echo "Error: No input water depth file specified in the WATER environment variable.";
	exit 1;
fi
if [[ -z "${RAINFALL}" ]]; then
	echo "Error: No input rainfall file specified in the RAINFALL environment variables.";
	exit 1;
fi

if [[ -z "${OUTPUT}" ]]; then
	echo "Error: No output directory specified in the OUTPUT environment variable.";
	exit 1;
fi

if [[ ! -r "${RAINFALL}" ]]; then
	echo "Error: That input rainfall file either doesn't exist, isn't a directory, or we don't have permission to access it.";
	exit 3;
fi
if [[ ! -r "${WATER}" ]]; then
	echo "Error: That input water depth file either doesn't exist, isn't a directory, or we don't have permission to access it.";
	exit 3;
fi

if [[ ! -d "${OUTPUT}" ]]; then
	mkdir "${OUTPUT}";
fi

export PATH=$HOME/software/bin:$PATH;


OUTPUT_UNIQ="${OUTPUT%/}_uniq"; # Stript trailing slash, if present
OUTPUT_TFRECORD="${OUTPUT%/}_tfrecord"; # Stript trailing slash, if present

mkdir -p "${OUTPUT_UNIQ}" "${OUTPUT_TFRECORD}";

echo ">>> Settings";

echo "RAINFALL $RAINFALL";
echo "WATER $WATER";
echo "OUTPUT $OUTPUT";
echo "COUNT_FILE $COUNT_FILE";
echo "ARGS $ARGS";

echo ">>> Installing requirements";
cd ../aimodel || { echo "Error: Failed to cd to ai model directory"; exit 1; };
conda run -n py38 pip install -r requirements.txt;
cd ../rainfallwrangler || { echo "Error: Failed to cd back to rainfallwrangler directory"; exit 1; };
npm install;
echo ">>> Converting dataset to .jsonl.gz";
/usr/bin/env time -v src/index.mjs recordify --verbose --rainfall "${RAINFALL}" --water "${WATER}" --output "${OUTPUT}" --count-file "${COUNT_FILE}" ${ARGS};
echo ">>> Deduplicating dataset";
# This also automatically recompresses for us - hence the source/target rather than in-place
srun --comment 'RainUniq' --exclusive -p compute --exclusive /usr/bin/env time -v src/index.mjs uniq --source "${OUTPUT}" --target "${OUTPUT_UNIQ}" --count-file "${COUNT_FILE}";
echo ">>> Removing intermediate output";
rm -r "${OUTPUT}";
echo ">>> Queuing .jsonl.gz → tfrecord";
INPUT="${OUTPUT_UNIQ}" OUTPUT="${OUTPUT_TFRECORD}" sbatch ./slurm-jsonl2tfrecord.job;
echo ">>> exited with code $?";
