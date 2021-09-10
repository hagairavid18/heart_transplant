# heart_transplant
This project is part of my participation in "Dkalim" program: a special program for honored students at Ben Gourion University. 
As part of the program, I worked with Professor Moshe Sipper which specializes in evolutionary algorithms and machine learning.
In this specific project, we worked in cooperation with two doctors, on a dataset of heart transplants from UNOS: United Network for Organ Sharing.

The purpose of this project is to help doctors in predicting the success of heart transplant of patients.
We defined success as surviving one year after the transplation date.We defined success as surviving one year after the transplation date.

In order to deal with this problem, we had to use traditional concepts of Survival Analsys: a collection of statistical procedures for data analysis where the outcome variable of interest is time until an event occurs.
For our purpose, the event will be the death of the patients.
In additional to the traditional Survival Analsys concepts, I used some combined algorithms of Machine learning and survival analsys to deal this problem. More specificaly, I used an algorithm called :"Gradient Boosting Survival Analysis".
This algorithm is intended to use the advandates of boosting methodes with adpations to survival anlasys loss function.
https://scikit-survival.readthedocs.io/en/stable/user_guide/boosting.html

The model i built, can predict the chances of patient to survive one year in a score of 0.7 AUC.

