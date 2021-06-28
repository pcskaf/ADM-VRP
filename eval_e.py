import pandas as pd
# from utils_demo import f_get_results_plot_seaborn, f_get_results_plot_plotly
from reinforce_baseline import load_tf_model
from utils import get_journey, read_from_pickle
from train import validate
from utils import get_cur_time
import pickle
import tensorflow as tf
from attention_dynamic_model import set_decode_type

import glob
import tsplib95
import argparse
import os
import pickle
import numpy as np
# import tensorflow as tf
# from utils.data_utils import check_extension, save_dataset

def transform_data(filename, node_coords, demand, capacity, graph_size):
    # num_samples=1
    # CAPACITIES = {
    #     10: 20.,
    #     20: 30.,
    #     50: 40.,
    #     100: 50.
    # }
    # seed=1234
    # depo1, graphs1, dem1 = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, 2), seed=seed),
    #                         tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2), seed=seed),
    #                         tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
    #                                                   dtype=tf.int32, seed=seed), tf.float32) / tf.cast(
    #                             40, tf.float32)
    #                         )
    # print(depo1)
    # print(graphs1)
    # print(dem1)
    # print(tf.data.Dataset.from_tensor_slices((list(depo1), list(graphs1), list(dem1))))

    depo = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    graphs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    dem = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

    if demand[0] == 0:
        # x=tf.stack(tf.cast(demand[1:], tf.float32) / tf.cast(capacity, tf.float32))
        depo.write(0, node_coords[0])
        x = 0
        graphs.write(x, node_coords[1:])
        x = 0
        dem.write(x,demand[1:])
        # dem.write(x, demand)
        # depo, graphs, demand = (node_coords[0], node_coords[1:], tf.cast(demand[1:], tf.float32) / tf.cast(capacity, tf.float32))
        # return (node_coords[0], node_coords[1:], tf.cast(demand[1:], tf.float32) / tf.cast(capacity, tf.float32))
    else:
        # x = tf.stack(tf.cast(demand[:(len(demand)-1)], tf.float32) / tf.cast(capacity, tf.float32))
        # print(x)
        depo.write(0, node_coords[len(node_coords)-1])
        x=0
        for i in range(len(node_coords)-1):
            # x=tf.stack([x1,demand[i+2]],0)
            graphs.write(x, node_coords[i])
            x += 1
        # print(graphs.stack())
        x = 0
        for i in range(len(demand)-1):
            # x=tf.stack([x1,demand[i+2]],0)
            dem.write(x, demand[i])
            x += 1
        # print(dem.stack())
        # depo, graphs, demand = (node_coords[0], node_coords[1:], tf.cast(demand[:(len(demand)-1)], tf.float32) / tf.cast(capacity, tf.float32))
        # return (node_coords[0], node_coords[1:], tf.cast(demand[:(len(demand)-1)], tf.float32) / tf.cast(capacity, tf.float32))

    # print(depo.stack())
    # print(graphs.stack())
    # print(dem.stack())
    print(tf.data.Dataset.from_tensor_slices((list(depo.stack()), list(graphs.stack()), list(dem.stack()))))
    return tf.data.Dataset.from_tensor_slices((list(depo.stack()), list(graphs.stack()), list(dem.stack())))
    # save_to_pickle('Validation_dataset_{}.pkl'.format(filename), (depo, graphs, demand))

def getbehiclesfromname(str):
    # print(str.split("-k")[1])
    return int(str.split("-k")[1])

def read_data_from_vrp(file):
    for f in glob.glob(file):
        filename = open(f, 'r')
        problem = tsplib95.load(f)

        coords=list(problem.node_coords.values())
        dem = list(problem.demands.values())

        # FOR E INSTANCES
        veh=getbehiclesfromname(problem.name)
        # depo, graphs, demand = transform_data(coords, dem, veh, problem.dimension - 1)

        # FOR M AND CMT INSTANCES
        # veh=problem.vehicles
        # depo, graphs, demand = transform_data(coords, dem, problem.vehicles, problem.dimension-1)

        return transform_data(problem.name, coords, dem, veh, problem.dimension - 1)

#CHANGES WERE MADE TO THE models.py AND distances.py SEARCH FOR "PERSONAL CHANGES"

def generate_data_onfly(num_samples=10000, graph_size=20):
    """Generate temp dataset in memory
    """

    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    depo, graphs, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, 2)),
                            tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2)),
                            tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
                                                      dtype=tf.int32), tf.float32)/tf.cast(CAPACITIES[graph_size], tf.float32)
                            )

    return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))

def read_from_pickle(path, return_tf_data_set=True, num_samples=None):
    """Read dataset from file (pickle)
    """

    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    # print(objects)
    objects = objects[0]
    if return_tf_data_set:
        depo, graphs, demand = objects
        # print(depo)
        # print(graphs)
        # print(demand)
        if num_samples is not None:
            return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand))).take(num_samples)
        else:
            return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))
    else:
        return objects

def main(input_path):
    MODEL_PATH = 'C:/Users/User/Documents/ΠΑΝΕΠΙΣΤΗΜΙΟ/Διπλωματική/Reinforcement Learning/data/ADM-VRP/model_checkpoint_epoch_99_VRP_20_2021-06-02.h5'
    # Create and save validation dataset
    # VAL_SET_PATH_vrp = 'C:/Users/User/Desktop/INSTANCES/M/M-n121-k7.vrp'
    # validation_dataset = read_data_from_vrp(VAL_SET_PATH_vrp)
    # VAL_SET_PATH = 'Validation_dataset_VRP_50_2021-06-24.pkl'
    # validation_dataset = read_from_pickle(VAL_SET_PATH)
    # print(get_cur_time(), 'validation dataset loaded')
    # print(validation_dataset)

    embedding_dim=128
    GRAPH_SIZE=100
    # Initialize model
    model_tf = load_tf_model(MODEL_PATH,
                             embedding_dim=embedding_dim,
                             graph_size=GRAPH_SIZE)
    set_decode_type(model_tf, "sampling")
    print(get_cur_time(), 'model loaded')

    # validate(validation_dataset, model_tf,1000)

    print("My data")
    # VAL_SET_PATH_vrp = 'C:/Users/User/Desktop/INSTANCES/Eilon/E-n22-k4.vrp'
    VAL_SET_PATH_vrp = input_path
    validation_dataset = read_data_from_vrp(VAL_SET_PATH_vrp)
    print(validation_dataset)

    validate(validation_dataset, model_tf,1000)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")

    main(parser.parse_args().datasets[0])
