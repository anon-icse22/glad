# parent of defects4j repo directories
import os
from shutil import which
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+'/data/defects4j-buggy-projects/'
# defects4j home directory
D4J_HOME = "/".join(which("defects4j").split("/")[:-3]) + "/"
# BPE info
BPE_OP_FILE = './etc_data/jmtrain_laxlit_v2_varpairs_BPE.pkl'
BPE_VOCAB_FILE = 'etc_data/jmtrain_laxlit_v2_vocab_BPE.pkl'
# where perfect fl information can be fetched
BUG_INFO_JSON = './etc_data/defects4j-bugs.json'
# where candidate patch files are generated
PATCH_DIR = os.path.dirname(os.path.abspath(__file__))+'/generated_patches/'

def repo_path(proj, bugid):
    return ROOT_DIR + f'{proj}_{bugid}/'

class TimeoutException(Exception):
    pass

def d4j_path_prefix(proj, bug_num):
    if proj == 'Chart':
        return 'source/'
    elif proj == 'Closure':
        return 'src/'
    elif proj == 'Lang':
        if bug_num <= 35:
            return 'src/main/java/'
        else:
            return 'src/java/'
    elif proj == 'Math':
        if bug_num <= 84:
            return 'src/main/java/'
        else:
            return 'src/java/'
    elif proj == 'Mockito':
        return 'src/'
    elif proj == 'Time':
        return 'src/main/java/'
    else:
        raise ValueError(f'Unrecognized project {proj}')

def d4j_test_path_prefix(proj, bug_num):
    if proj == 'Chart':
        return 'tests/'
    elif proj == 'Closure':
        return 'test/'
    elif proj == 'Lang':
        if bug_num <= 35:
            return 'src/test/java/'
        else:
            return 'src/test/'
    elif proj == "Math":
        if bug_num <= 84:
            return 'src/test/java/'
        else:
            return 'src/test/'
    elif proj == 'Mockito':
        return 'test/'
    if proj == "Time":
        return 'src/test/java/'
    else:
        raise ValueError(f'Cannot find test path prefix for {proj}{bug_num}')

def parse_abs_path(jfile):
    repo_dir_name = jfile.removeprefix(ROOT_DIR).split('/')[0]
    repo_dir_path = ROOT_DIR + repo_dir_name + '/'
    rel_jfile_path = jfile.removeprefix(repo_dir_path)
    return repo_dir_path, rel_jfile_path

def log(*args):
    now = datetime.now()
    now_str = now.strftime(r'%Y-%m-%d %H:%M:%S.%f')
    print(f'[{now_str}]', *args, flush=True)