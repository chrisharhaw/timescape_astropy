import pytest
import sys
sys.path.append('/Users/christopherharvey-hawes/Documents/Physics/timescape_astropy')
import timescape

#Should replace the sys.path.append with an __init__.py file in the timescape folder
#Need to make test functions for all the functions in timescape.py, starting with test_*function_name*()


ts = timescape.timescape()

# def test_hubble_distance():
#     assert ts.hubble_distance()== 

def test_Om_bare():
    assert ts.Om_bare(0)== ts.Om0_bare

def test_Ok_bare():
    assert ts.Ok_bare(0)== ts.Ok0_bare

def test_OQ_bare():
    assert ts.OQ_bare(0)== ts.OQ0_bare

def test_Om_dressed():
    assert ts.Om_dressed(0)== ts.Om0_dressed

def test_Ok_dressed():
    assert ts.Ok_dressed(0)== ts.Ok0_dressed

def test_OQ_dressed():
    assert ts.OQ_dressed(0)== ts.OQ0_dressed

# def test_wall_time():
#     assert ts.wall_time(0)== #value
#     assert ts.wall_time(1)== #value

