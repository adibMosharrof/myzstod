from argparse import REMAINDER, ArgumentParser

import numpy as np
def parse_args():
    """
    Helper function parsing the command line options.
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=(
            "Habana Gaudi distributed training launch helper utility that will spawn up multiple distributed"
            " processes."
        )
    )

    # Optional arguments for the launch helper
    parser.add_argument("--num_nodes", type=int, default=2, help="Number of nodes to use")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs per node")
    parser.add_argument("--hostnames_path", type=str, default="test.txt", help="Number of GPUs per node")


    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.hostnames_path,'r') as fr:
        lines = fr.readlines()
    lines = np.unique(lines)
    with open('myhostfile','a') as fw:
        for l in lines:
            fw.write(f"{l.strip()} slots={args.num_gpus}\n")
            
    master_addr = lines[0].strip()
    with open('.deepspeed_env','a') as f:
        f.write(f"MASTER_ADDR={master_addr}\n")
        f.write(f"MASTER_PORT=6000\n")
        f.write(f"OMP_NUM_THREADS=16\n")
        f.write(f"MKL_NUM_THREADS=16\n")
        f.write(f"NCCL_IB_DISABLE=1\n")
        f.write(f"NCCL_IBEXT_DISABLE=1\n")

if __name__ == "__main__":
    main()