from hashlib import sha256
from Crypto.PublicKey import RSA

from sklearn.datasets import load_iris, load_breast_cancer

import pandas as pd
import numpy as np

from utils import data_utils


def test_simple_hash():
    msg = 'hello world'.encode('utf-8')
    h = sha256(msg).digest()
    h_hex = sha256(msg).hexdigest()

    print(h == h_hex, type(h) == type(h_hex), h, h_hex)


def test_simple_encrypt():
    kp = RSA.generate(bits=1024)
    modulus = kp.n
    priv_key = kp.d
    pub_key = kp.e

    msg = 'hello world'.encode('utf-8')
    h = int.from_bytes(sha256(msg).digest(), byteorder='big')
    signature = pow(h, pub_key, modulus)
    print("Signature:", hex(signature))

    hash_from_signature = pow(signature, priv_key, modulus)
    print("Signature valid:", h == hash_from_signature)


def test_data_load_sklearn():
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    print(np.count_nonzero(breast_cancer.target))
    # print(df.columns)
    print(df.head())


def test_data_load_local():
    df = pd.read_csv("../data/breast_cancer_wisconsin.csv")
    # print(df.columns)
    print(df.head())


def test_data_load():
    train, test = data_utils.load(prop_train=.8, prop_test=.2)
    print(train.head())
    print(test.head())
