import hello_world
import rnn
import numpy as np
import utils

def main():
    p = hello_world.Person("Kevin")
    p.Greet()

    params = rnn.init_params(3,4,5)
    # rnn.rnn_cell(params)


    # TODO: Add this as a test
    values = [2,3,4,5]
    softmax = utils.softmax(values)
    print("Softmax output: ", softmax)
    print("Sum of Softmax Values: ", np.sum(softmax))

if __name__ == "__main__":
    main()