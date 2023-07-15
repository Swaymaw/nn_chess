import os
import h5py
import chess.pgn
from state import State
import numpy as np


def get_dataset(num_samples=None):
    x, y = [], []
    gn = 0
    files = os.listdir('data')
    files_n = len(files)
    values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}

    # pgn files in the data folder
    for fn in files:
        # fn = files[fn_n] for smaller data
        pgn = open(os.path.join('data', fn))
        while True:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                break
            res = game.headers['Result']
            if res not in values:
                continue
            value = values[res]
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                ser = State(board).serialize()
                x.append(ser)
                y.append(value)

            print(f'On file {fn}, parsing game {gn}, got {len(x)} examples.')
            if num_samples is not None and len(x) > num_samples:
                x = np.array(x)
                y = np.array(y)
                return x, y
            gn += 1
    x = np.array(x)
    y = np.array(y)
    return x, y


if __name__ == '__main__':
    X, Y = get_dataset(1e7)
    np.savez('processed/dataset_10M.npz', X, Y)
