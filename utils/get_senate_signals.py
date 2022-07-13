"""
This file load the signals from the dataset senate networks 
"""

import numpy as np
import pandas as pd

FILE_PATH = '../data/senate_data/'

N_STATES = 50
DEM_CODE = 100
REP_CODE = 200
INDEP_CODE = 328

remove_votes = True


def get_signals(number, file_path=''):
    """
    Load the data from the senate network dataset with the number specified as
    an argument, and it returns the graph signals corresponding to the
    different votations and the labels indicating the parting to which each
    node belongs. Information about the dataset is available at
    https://voteview.com/
    """
    print('Loading dataset for senate', number)

    # Prepare votes data
    votes_file = '{}S{}_votes.csv'.format(file_path, number)
    votes = np.genfromtxt(votes_file, delimiter=',', skip_header=1, usecols=(2, 3, 4))
    n_votes = int(votes[:,0].max())
    votes[votes[:,2] == 1,2] = 1 
    votes[votes[:,2] == 6,2] = -1
    votes[votes[:,2] == 9,2] = 0  

    # Prepare states party (node labels) data
    members_file = '{}S{}_members.csv'.format(file_path, number)
    members = pd.read_csv(members_file)
    if members['state_abbrev'][0] == 'USA':
        members = members.drop(index=0)
        print('Dropping president from senators list')

    members['party_code'].loc[members['party_code'] == DEM_CODE] = 'D'
    members['party_code'].loc[members['party_code'] == REP_CODE] = 'R'
    members['party_code'].loc[members['party_code'] == INDEP_CODE] = 'I'

    # Get states id
    states_icpsr = members['state_icpsr'].unique()
    # In case you need to get the indixes sorted for some reason
    # states_icpsr = np.unique(members['state_icpsr'].to_numpy())
    N = int(states_icpsr.shape[0])

    assert N == N_STATES, 'Error in reading states'

    # Get node labels by checking party of the senators
    node_labels = []
    for i, state_icpsr in enumerate(states_icpsr):
        parties = members['party_code'].loc[members['state_icpsr']==state_icpsr]
        parties = np.unique(parties.values)
        node_labels.append(parties[0] if len(parties) == 1 else 'M')

    # Get vote signals
    X = np.zeros((N, n_votes))
    rm_votations = []
    for i in range(n_votes):
        votes_i = votes[votes[:,0] == (i+1),:]

        for j, state_icpsr in enumerate(states_icpsr):
            icpsr = members['icpsr'].loc[members['state_icpsr']==state_icpsr].to_numpy()
            casted_votes = votes_i[np.isin(votes_i[:,1], icpsr), 2]
            X[j,i] = np.sum(casted_votes)

            if not np.isin(casted_votes, [1, -1, 0]).all():
                rm_votations.append(i)

    rm_votations = np.unique(np.array(rm_votations))
    print('Votes different that 1,-1, and 0 in votation rolls:')
    print(rm_votations)

    print('Total number of votes:', n_votes)
    if remove_votes:
        X = np.delete(X, rm_votations, axis=1)
    print('Votes after prepocessing:', X.shape[1])

    return X, node_labels

if __name__ == '__main__':
    senate_number = 115
    X, labels = get_signals(senate_number, FILE_PATH)
    np.save(FILE_PATH + 'X' + str(senate_number), X)
    np.save(FILE_PATH + 'labels' + str(senate_number), labels)
    print('Senate', str(senate_number), 'saved at', FILE_PATH)
