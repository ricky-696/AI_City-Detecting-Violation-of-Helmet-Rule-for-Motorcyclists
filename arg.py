import argparse

def arg():
    parser = argparse.ArgumentParser()
    #沒用到
    #parser.add_argument('--dirpath', type=str, default='/workspaces/mvl/ai-jam/program', help='image source')#, required=True)
    #測試
    #parser.add_argument('--inputpath', type=str, default='/home/hhc102u/ai-jam/AICITY_Detection/test2_output', help='image source')#, required=True)
    #parser.add_argument('--outputpath', type=str, default='/home/hhc102u/ai-jam/AICITY_Detection/test2_cutter_output', help='image source')#, required=True)
    #正式
    parser.add_argument('--start', type=int, help='image source')#, required=True)
    parser.add_argument('--end', type=int, help='image out')#, required=True)

    #parser.add_argument('--inputpath', type=str, default='/workspaces/mvl/ai-jam/program/prod/10.16_0/input', help='image source')#, required=True)
    #parser.add_argument('--outputpath', type=str, default='/workspaces/mvl/ai-jam/program/prod/10.16_0/output', help='image out')#, required=True)


    args = parser.parse_args()
    
    return args