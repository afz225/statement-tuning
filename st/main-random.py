import argparse
import json
import logging
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import numpy as np
from pathlib import Path

import pandas as pd

import random

from .create_data import generate_train_data
from .train import train_st
from .eval import run_eval

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--mode", "-m", type=str, default="train", help="Mode: train or evaluate"
    )
    parser.add_argument(
        "--data",
        "-d",
        default=None,
        type=str,
        help="Path to huggingface dataset locally or on the hub.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="baseline",
        help="Experiment name for saving model and results",
    )
    parser.add_argument(
        "--results",
        "-r",
        type=str,
        default="../results/",
        help="Path to directory output results file for evaluation.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of Runs",
    )
    parser.add_argument(
        "--model",
        default='roberta-base',
        type=str,
        help="Base Encoder Model to use",
    )
    parser.add_argument(
        "--model_save_path", type=str, default="../models/", help="Folder to save models"
    )
    parser.add_argument(
        "--lr",
        default=1e-06,
        type=float,
        help="Learning rate, e.g. 1e-06",
    )
    parser.add_argument(
        "--max_epochs",
        default=1000,
        type=float,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--patience",
        default=10,
        type=float,
        help="Training patience",
    )
    parser.add_argument(
        "--tolerance",
        default=20,
        type=int,
        help="Tolerance of statement size",
    )
    parser.add_argument(
        "--spc",
        type=int,
        default=4000,
        metavar="N",
        help="Acceptable values are N, where N is an integer. Default 4000.",
    )
    parser.add_argument(
        "--ppc",
        type=int,
        default=5,
        metavar="N",
        help="Acceptable values are N, where N is an integer. Default 5.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        type=str,
        help="Datasets to exclude from the training pool.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to hugging face cache.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=16,
        metavar="N",
        help="Acceptable values are N, where N is an integer. Default 16.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=514,
        metavar="N",
        help="Acceptable values are N, where N is an integer. Default 32.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="If True, skip evaluation after training.",
    )
    parser.add_argument(
        "--n_shots",
        type=str,
        default='0',
        help="List of n-shots",
    )
    return parser

def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    check_argument_types(parser)
    return parser.parse_args()
def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(
                    f"Argument '{action.dest}' doesn't have a type specified."
                )
            else:
                continue

def agg_std(std_list):
    # print(len(std_list), std_list)
    return (sum([std**2 for std in std_list])/len(std_list))**0.5

def main(args: Union[argparse.Namespace, None] = None) -> None:
    dirname = os.path.dirname(__file__)
    if not args:
        parser = setup_parser()
        args = parse_eval_args(parser)
        
    if args.mode == "train":
        excluded_data = [] if args.exclude is None else args.exclude.split(',')
        for i in range(args.n_runs):
            train_data = generate_train_data(
                excluded_data=excluded_data,
                ppc=args.ppc,
                spc=args.spc,
                cache_dir=args.cache_dir,
            )
            print(train_data['train']['is_true'])
            tokenizer, model, trainer = train_st(
                     model_name=args.model,
                     tolerance=args.tolerance,
                     data=train_data,
                     batch_size=args.batch_size,
                     lr=args.lr,
                     patience=args.patience,
                     n_epochs=args.max_epochs,
                     context_len=args.context_len,
                    )
            
            models_path = os.path.join(dirname, args.model_save_path)
            models_list = [model[-1] for model in os.listdir(models_path) if args.experiment_name+"-" in model]
            model_number = 0 if models_list == [] else int(max((models_list)))+1
            
            save_path = os.path.join(dirname, args.model_save_path, args.experiment_name+f"-{model_number}")
            
            trainer.save_model(save_path)
            tokenizer.save_pretrained(save_path)

            if args.skip_eval == False:
                results = []
                
                for shot in args.n_shots.split(','):
                    if shot != 'full':
                        shot = int(shot)
                    runs = []
                    speeds=[]
                    for j in range (args.n_runs):
                        run_speeds, data_accuracies = run_eval(
                            tokenizer=tokenizer, 
                            model=model, 
                            batch_size=args.batch_size, 
                            cache_dir=args.cache_dir, 
                            n_shot=0,
                        )
                        runs.append(data_accuracies)
                        speeds.append(run_speeds)
                        
                    df = pd.DataFrame(runs)
                    speed_df = pd.DataFrame(speeds)
                    result_df = pd.DataFrame()
                    result_df['Mean'] = df.mean(axis=0)
                    result_df['std'] = df.std(axis=0)
                    result_df['speed'] = speed_df.mean(axis=0)
                    result_df = result_df.reset_index()
                    result_df = result_df.rename(columns={'index': 'dataset'})
                    result_df['shot'] = shot
                    result_df['model'] = args.model
                    result_df['PPC'] = args.ppc
                    result_df['SPC'] = args.spc
                    result_df['excluded'] = "None" if args.exclude is None else args.exclude
                    
                    results.append(result_df)
            
                results = pd.concat(results, ignore_index=True)
                agg_results = pd.concat([results.groupby(['model', 'shot', 'SPC', 'PPC', 'excluded', 'dataset'])['Mean'].mean().reset_index(), results.groupby(['model', 'shot', 'SPC', 'PPC', 'excluded', 'dataset'])['std'].agg(agg_std).reset_index()['std'], results.groupby(['model', 'shot', 'SPC', 'PPC', 'excluded', 'dataset'])['speed'].mean().reset_index()['speed']], axis=1)
                results_save_path = os.path.join(dirname, args.results, args.experiment_name+".csv")
                agg_results.to_csv(results_save_path)
            

            
                
        # print(train_data)
        
    elif args.mode == "evaluate":
        # runs = []
        # for j in range (2):
        #     runs.append(run_eval(
        #         tokenizer=tokenizer, 
        #         model=model, 
        #         batch_size=args.batch_size, 
        #         cache_dir=args.cache_dir, 
        #         n_shot=32,
        #     ))
        results = []
        for i in range(args.n_runs):
            model_path = os.path.join(dirname, args.model_save_path, args.experiment_name+f"-{i}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=args.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            for shot in args.n_shots.split(','):
                if shot != 'full':
                    shot = int(shot)
                
                runs = []
                speeds = []
                for j in range (args.n_runs):
                    run_speeds, data_accuracies = run_eval(
                        tokenizer=tokenizer, 
                        model=model, 
                        batch_size=args.batch_size, 
                        cache_dir=args.cache_dir, 
                        n_shot=0,
                    )
                    runs.append(data_accuracies)
                    speeds.append(run_speeds)
                df = pd.DataFrame(runs)
                speed_df = pd.DataFrame(speeds)
                result_df = pd.DataFrame()
                result_df['Mean'] = df.mean(axis=0)
                result_df['std'] = df.std(axis=0)
                result_df['speed'] = speed_df.mean(axis=0)
                result_df = result_df.reset_index()
                result_df = result_df.rename(columns={'index': 'dataset'})
                result_df['shot'] = shot
                result_df['model'] = args.model
                result_df['PPC'] = args.ppc
                result_df['SPC'] = args.spc
                result_df['excluded'] = "None" if args.exclude is None else args.exclude
                
                results.append(result_df)    
        
        results = pd.concat(results, ignore_index=True)
        agg_results = pd.concat([results.groupby(['model', 'shot', 'SPC', 'PPC', 'excluded', 'dataset'])['Mean'].mean().reset_index(), results.groupby(['model', 'shot', 'SPC', 'PPC', 'excluded', 'dataset'])['std'].agg(agg_std).reset_index()['std'], results.groupby(['model', 'shot', 'SPC', 'PPC', 'excluded', 'dataset'])['speed'].mean().reset_index()['speed']], axis=1)
        results_save_path = os.path.join(dirname, args.results, args.experiment_name+"-shuffled.csv")
        agg_results.to_csv(results_save_path)
        # print(agg_results)
        # df = pd.DataFrame(runs)
        # result_df = pd.DataFrame()
        # result_df['Mean'] = df.mean(axis=0)
        # result_df['std'] = df.std(axis=0)
        # result_df = result_df.reset_index()
        # result_df = result_df.rename(columns={'index': 'dataset'})
        # print(result_df)

if __name__ == "__main__":
    main()