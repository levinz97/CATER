import os
import shutil
import argparse

def get_parser():
    start = 5200
    end = 5214
    source_path = os.path.join('.','raw_data','raw_data_from_005200_to_005699_sort', '005200-005299_sort')
    target_path = os.path.join('.', 'dataset', 'images', 'image')
    parser = argparse.ArgumentParser(description="copy the image to dataset")
    parser.add_argument("--destination", '-d', default=target_path, dest='target_path')
    parser.add_argument("--source", '-s', default=source_path, dest='source_path')
    parser.add_argument("--first", '-f', default=start, type=int, dest='start')
    parser.add_argument("--last", '-l', default=end, type=int, dest='end')
    parser.add_argument("--dry-run", default=False, action='store_true', help='Perform a trial copy with no changes made')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_parser()

    IMG_EXTENSIONS = ['.png','.jpg']
    is_image_file = lambda filename : any(filename.endswith(ext) for ext in IMG_EXTENSIONS)
    num_of_file = 0
    for d in sorted(os.listdir(args.source_path)):
        if int(d[-6:]) in range(args.start, args.end+1):
            for root, _, fnames in sorted(os.walk(os.path.join(args.source_path,d))):
                for fn in fnames:
                    if not is_image_file(fn):
                        continue
                    else:
                        if not os.path.exists(os.path.join(args.target_path, fn)):
                            f = os.path.join(root,fn)
                            num_of_file += 1
                            if not args.dry_run:
                                shutil.copy2(f, args.target_path)
                            else:
                                print(f"copy {fn} to  {args.target_path}")
    print(f"copy {num_of_file} files from {args.source_path} to {args.target_path}")
    print("dry run, no effect" if args.dry_run else " ")
