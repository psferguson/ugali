"""
This is the Ultra-faint Galaxy Likelihood (UGaLi) software package.
"""
__author__ = "Keith Bechtol & Alex Drlica-Wagner"
__email__ = "bechtol@kicp.uchicago.edu, kadrlica@fnal.gov"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# ADW: Is this a good idea?
import ugali.analysis.isochrone as isochrone
import ugali.analysis.kernel as kernel
import ugali.analysis.source as source
