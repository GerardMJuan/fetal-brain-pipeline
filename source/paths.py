"""
Various paths
"""

import os

# global paths
SINGULARITY_twai = "/home/gmarti/SINGULARITY/twai_latest.sif"
SINGULARITY_bounti = "/home/gmarti/SINGULARITY/bounti.sif"
SINGULARITY_dhcp = (
    # "/home/gmarti/SINGULARITY/dhcp-structural-pipeline_fetal-custom.sif"
    "/home/gmarti/SINGULARITY/dhcp-pipeline-multifact.sif"
)

env = {
    **os.environ,
}
