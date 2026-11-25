#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}/pointnet2_ops_lib"
python setup.py install

cd "${ROOT_DIR}/Chamfer3D"
python setup.py install
