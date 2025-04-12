{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8dda3887",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np #Enables efficient array operation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2bf4364d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = np.array([[1,1],[1,2],[1,3],[1,4]])\n",
    "y = np.array([2,3,4,5])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d03139d7",
   "metadata": {},
   "source": [
    "$\\beta = (X^TX)^{-1}X^Ty$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cd809b7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "beta = np.linalg.inv(X.T @ X) @ X.T @ y"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a105d03b",
   "metadata": {},
   "source": [
    "$\\hat{y} = X\\beta = X(X^{T}X)^{-1}X^{T}y$ (approximation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "305714b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = X @ beta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7fbf51bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Weights: [1. 1.]\n",
      "Predictions: [2. 3. 4. 5.]\n"
     ]
    }
   ],
   "source": [
    "print(\"Weights:\", beta)\n",
    "print(\"Predictions:\", y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be48512b",
   "metadata": {},
   "source": [
    "- The normal equation is great for when the dataset is small\n",
    "- Also great for finding the exact solution (no need for iteration)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
