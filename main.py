from block import Block
from blockchain import Blockchain
from device import Device, DevicesInNetwork

import os
import sys
import argparse
import random
import time
from datetime import datetime

date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_folder_path = f"logs/{date_time}"
model_snapshots_folder = "models"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='BCFL_kmeans_Simulation')

# debug attributes
parser.add_argument('-v', '--verbose', type=int, default=1, help='print verbose debug log')

# FL attributes
# parser.add_argument('-B', '--batchsize', type=int, default=10, help='batch size used in local training')
parser.add_argument('-lr', '--learning_rate', type=float, default=.01, help='learning rate of k-means')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communication rounds')
parser.add_argument('-nd', '--num_devices', type=int, default=20, help='number of devices in the simulation')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help='number of malicious devices')
parser.add_argument('-iid', '--IID', type=int, default=0, help='whether to allocate data in iid setting')
parser.add_argument('-gc', '--num_global_centroids', type=int, default=5, help='number of centroids in the globally '
                                                                               'trained model')
parser.add_argument('-lc', '--num_local_centroids', type=int, default=3, help='number of centroids in locally trained '
                                                                              'models')
parser.add_argument('-le', '--num_local_epochs', type=int, default=1, help='number of epochs to perform for the '
                                                                           'acquisition of local update')

# BC attributes (consensus & committee parameters)
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=6, help='amount of rounds after which to kick a '
                                                                           'malicious data owner')
parser.add_argument('-lo', '--lazy_knock_out_rounds', type=int, default=10, help='amount of rounds after which to '
                                                                                 'kick a non-responsive data owner')
parser.add_argument('-vh', '--validator_threshold', type=float, default=1.0, help='threshold value of a difference in '
                                                                                  'accuracy with which to identify '
                                                                                  'malicious data owners')
parser.add_argument('-pow', '--pow_difficulty', type=int, default=0, help='difficulty of proof-of-work')

# Additional BC attributes (to make entire process more efficient)
parser.add_argument('-cmt', '--committee_member_wait_time', type=float, default=0.0, help="default time window during "
                                                                                          "which committee members "
                                                                                          "wait for local updates to "
                                                                                          "be sent. Wait time of 0.0 "
                                                                                          "is associated with no time "
                                                                                          "limit.")
parser.add_argument('-cmh', '--committee_member_threshold', type=float, default=1.0, help="threshold value for the "
                                                                                          "difference in performance"
                                                                                          " to determine whether to "
                                                                                          "consider a local update")

# distributed system attributes
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='the odds of a device being and '
                                                                                'staying online')
parser.add_argument('-els', '--equal_link_speed', type=int, default=1, help='used to simulate transmission delay. If '
                                                                            'set to 1, every device has equal link '
                                                                            'speed (bytes/sec). If set to 0, '
                                                                            'link speed is determined randomly.')
parser.add_argument('-dts', '--data_transmission_speed', type=float, default=70000.0, help="volume of data that is "
                                                                                           "transmitted per second "
                                                                                           "when -els == 1.")
parser.add_argument('-ecp', '--equal_computation_power', type=int, default=1, help='used to simulation computation '
                                                                                   'power. If set to 1, every device '
                                                                                   'has equal computation power. If '
                                                                                   'set to 0, computation power is '
                                                                                   'determined randomly.')

# simulation attributes
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help='number of devices assigned to the roles '
                                                                            'of data owner, committee member and '
                                                                            'committee leader respectively')
parser.add_argument('-cs', '--check_signature', type=int, default=1, help='whether to check signatures, used to save '
                                                                          'time or to assume trust')
parser.add_argument('-aio', '--all_in_one_network', type=int, default=1, help='whether to have all devices be aware '
                                                                              'of and connected to each other device '
                                                                              ' in the network')

if __name__ == '__main__':

    # get arguments
    args = parser.parse_args()
    args = args.__dict__

    # set up from scratch
    # 0. create directory for log files
    if not os.path.isdir(log_folder_path):
        os.mkdir(log_folder_path)

    # 1. save arguments and parameters used
    with open(f'{log_folder_path}/args_used.txt', 'w') as f:
        f.write("Command line arguments used -\n")
        f.write(' '.join(sys.argv[1:]))
        f.write("\n\nAll arguments used -\n")
        for arg_name, arg in args.items():
            f.write(f'\n--{arg_name} {arg}')

    # 2. create folder to save models in, if it does not already exist
    if not os.path.isdir(model_snapshots_folder):
        os.mkdir(model_snapshots_folder)
    os.mkdir(f'{model_snapshots_folder}/{date_time}')

    # 3. get number of roles required in the network
    roles_requirement = args['hard_assign'].split(',')
    # determine roles to assign
    data_owners_needed = int(roles_requirement[0])
    committee_members_needed = int(roles_requirement[1])
    leaders_needed = int(roles_requirement[2])

    # 4. check arguments eligibility
    num_devices = args['num_devices']
    num_malicious = args['num_malicious']

    if num_devices < data_owners_needed + committee_members_needed + leaders_needed:
        sys.exit("ERROR: Roles assigned to devices exceed the maximum number of devices.")

    if num_devices < 3:
        sys.exit("ERROR: There are not enough devices in the network to execute. \n There need to be at least one "
                 "data owner, one committee member and one committee leader.")

    if num_malicious:
        if num_malicious > num_devices:
            sys.exit("ERROR: The number of malicious devices cannot exceed the total number of devices.")
        else:
            print(f'Malicious nodes and total devices set to {num_malicious}:{num_devices}')

    # 5. create devices in the network

    # 6. register devices and initialize global parameters

    # 7. build log files, to be filled during execution

    # BCFL-KMeans starts here
    # ...

    # i. obtain most recent block

    # ii. perform local learning step and share result with associated committee member

    # iii. committee members validate retrieved updates and aggregate viable results

    # iv. committee members send updated centroids to every leader

    # v. leaders build candidate blocks using the obtained centroids and send it to committee members for approval

    # vi. committee members vote on candidate blocks by sending their vote to all committee members (signed)

    # vii. leader that obtained a majority vote append their block to the chain and broadcast it to all devices
