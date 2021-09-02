import json
import util

def get_d4j_info_dict(json_file):
    with open(json_file) as f:
        info_list = json.load(f)
    bugname2info = {(x['project'], x['bugId']): x for x in info_list}
    return bugname2info

def get_failing_test_names(proj, bugid, info_file=util.BUG_INFO_JSON):
    bugname2info = get_d4j_info_dict(info_file)
    target_bug_info = bugname2info[(proj, bugid)]
    fail_tests_info = target_bug_info['failingTests']
    return [e['className'].strip()+'::'+e['methodName'] for e in fail_tests_info]

def get_true_buggy_locs(proj, bugid, info_file=util.BUG_INFO_JSON):
    bugname2info = get_d4j_info_dict(info_file)
    target_bug_info = bugname2info[(proj, bugid)]
    buggy_files = list(target_bug_info['changedFiles'].keys())
    buggy_lines = [list(target_bug_info['changedFiles'][bf].values()) for bf in buggy_files]
    # get purified location:
    true_fix_locations = []
    for loc_list in buggy_lines:
        linear_loc_list = sorted([e[0] for e in loc_list[0]])
        tfl_for_file = []
        for idx, loc in enumerate(linear_loc_list):
            if linear_loc_list[idx-1] == loc-1:
                continue
            else:
                tfl_for_file.append(loc)
        true_fix_locations.append(tfl_for_file)
    return buggy_files, true_fix_locations

if __name__ == '__main__':
    print(get_true_buggy_locs('Chart', 14))