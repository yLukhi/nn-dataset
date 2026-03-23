from typing import Optional

import ab.nn.util.db.Read as DB_Read
import ab.nn.util.Train as Train
from ab.nn.util.Const import default_epoch_limit_minutes
from pandas import DataFrame
import functools

from ab.nn.util.db.Query import JoinConf


@functools.lru_cache(maxsize=10)
def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None, max_rows=None, sql: Optional[JoinConf] = None, nn_prefixes=None,
         unique_nn=False, nn_total_layers=False, nn_max_depth=False, nn_has_residual=False, nn_flops=False, nn_dropout_count=False, nn_total_params=False, nn_has_attention=False,
         prm__dropout=False, prm__momentum=False, prm__lr=False, prm__weight_decay=False, prm__batch=False) -> DataFrame:
    """
    Get the NN model code and all related statistics as a pandas DataFrame.

    For the detailed description of arguments see :ref:`ab.nn.util.db.Read.data()`.

    Parameters:
      - only_best_accuracy (bool): If True, for each unique combination of
          (task, dataset, metric, nn, epoch) only the row with the highest accuracy is returned.
          If False, all matching rows are returned.
      - task, dataset, metric, nn, epoch: Optional filters to restrict the results.
      - max_rows (int): Specifies the maximum number of results.
      - nn_total_layers (bool): If True, include 'nn_total_layers' column in results.
      - nn_max_depth (bool): If True, include 'nn_max_depth' column in results.
      - nn_has_residual (bool): If True, include 'nn_has_residual' column in results.
      - nn_flops (bool): If True, include 'nn_flops' column in results.
      - nn_dropout_count (bool): If True, include 'nn_dropout_count' column in results.
      - nn_total_params (bool): If True, include 'nn_total_params' column in results.
      - nn_has_attention (bool): If True, include 'nn_has_attention' column in results.
      - prm__dropout (bool): If True, include 'prm_dropout' column extracted from prm dict.
      - prm__momentum (bool): If True, include 'prm_momentum' column extracted from prm dict.
      - prm__lr (bool): If True, include 'prm_lr' column extracted from prm dict.
      - prm__weight_decay (bool): If True, include 'prm_weight_decay' column extracted from prm dict.
      - prm__batch (bool): If True, include 'prm_batch' column extracted from prm dict.

    Returns:
      - A pandas DataFrame where each row is a dictionary containing:
          'task', 'dataset', 'metric', 'metric_code',
          'nn', 'nn_code', 'epoch', 'accuracy', 'duration',
          'prm', and 'transform_code'.

        If any of the nn_* parameters are True, the corresponding columns are included:
          'nn_total_layers', 'nn_max_depth', 'nn_has_residual',
          'nn_flops', 'nn_dropout_count', 'nn_total_params', 'nn_has_attention'

        If any of the prm__* parameters are True, the corresponding columns are included:
          'prm_dropout', 'prm_momentum', 'prm_lr', 'prm_weight_decay', 'prm_batch'
    """
    dt: tuple[dict, ...] = DB_Read.data(only_best_accuracy, task=task, dataset=dataset, metric=metric, nn=nn, epoch=epoch, max_rows=max_rows,
                                        sql=sql, nn_prefixes=nn_prefixes, unique_nn=unique_nn, nn_total_layers=nn_total_layers, nn_max_depth=nn_max_depth, 
                                        nn_has_residual=nn_has_residual, nn_flops=nn_flops, nn_dropout_count=nn_dropout_count, nn_total_params=nn_total_params, 
                                        nn_has_attention=nn_has_attention)
    df = DataFrame.from_records(dt)

    # Extract requested prm fields into separate columns
    prm_fields = {
        'dropout': prm__dropout,
        'momentum': prm__momentum,
        'lr': prm__lr,
        'weight_decay': prm__weight_decay,
        'batch': prm__batch,
    }
    if 'prm' in df.columns:
        for field, enabled in prm_fields.items():
            if enabled:
                df[f'prm_{field}'] = df['prm'].apply(lambda p, f=field: p.get(f) if isinstance(p, dict) else None)

    return df


@functools.lru_cache(maxsize=10)
def run_data(model_name=None, device_type=None, max_rows=None) -> DataFrame:
    """
    Get comprehensive runtime and tflite analytics as a pandas DataFrame.
    
    Combines analytics from both the 'run' table (device performance metrics) and 
    'tflite' table (model quantization metrics) joined by model_name. This enables
    analysis of how a tflite model performs across different devices.

    Parameters:
      - model_name (str | None): filter by model name (FK to nn.name)
      - device_type (str | None): filter by device type (only applies to run table)
      - max_rows (int | None): maximum number of results

    Returns:
      - A pandas DataFrame with columns from both tables:
      
        ** From run table (device runtime analytics) **
        'id', 'model_name', 'device_type', 'os_version', 'valid', 'emulator', 'error_message', 
        'duration', 'iterations', 'unit', 'cpu_duration', 'cpu_min_duration', 'cpu_max_duration', 
        'cpu_std_dev', 'cpu_error', 'gpu_duration', 'gpu_min_duration', 'gpu_max_duration', 
        'gpu_std_dev', 'gpu_error', 'npu_duration', 'npu_min_duration', 'npu_max_duration', 
        'npu_std_dev', 'npu_error', 'total_ram_kb', 'free_ram_kb', 'available_ram_kb', 'cached_kb',
        'in_dim_0', 'in_dim_1', 'in_dim_2', 'in_dim_3', 'device_analytics', 'precision_type'
        
        ** From tflite table (model metrics) **
        'tflite_id', 'tflite_accuracy', 'tflite_transform', 'tflite_precision_type'
        
    Note: Each row represents a device-specific run of a tflite model. Multiple runs of the 
    same tflite model on different devices will appear as separate rows. In the future, the 
    tflite table structure will remain stable while run data will expand to cover more devices.
    """
    # Get run data
    run_recs: tuple[dict, ...] = DB_Read.run_data(
        model_name=model_name, 
        device_type=device_type, 
        max_rows=max_rows
    )
    
    # Get tflite data
    tflite_recs: tuple[dict, ...] = DB_Read.tflite_data(
        model_name=model_name, 
        max_rows=max_rows
    )
    
    # Create a mapping of model_name to tflite data
    tflite_map = {rec['model_name']: rec for rec in tflite_recs}
    
    # Merge run data with tflite data
    merged_records = []
    for run_rec in run_recs:
        merged_rec = dict(run_rec)
        
        # Join with tflite data if available
        if merged_rec['model_name'] in tflite_map:
            tflite_rec = tflite_map[merged_rec['model_name']]
            merged_rec['tflite_id'] = tflite_rec.get('id')
            merged_rec['tflite_accuracy'] = tflite_rec.get('accuracy')
            merged_rec['tflite_transform'] = tflite_rec.get('transform')
            merged_rec['tflite_precision_type'] = tflite_rec.get('precision_type')
        else:
            # If no tflite data, set to None
            merged_rec['tflite_id'] = None
            merged_rec['tflite_accuracy'] = None
            merged_rec['tflite_transform'] = None
            merged_rec['tflite_precision_type'] = None
        
        merged_records.append(merged_rec)
    
    return DataFrame.from_records(merged_records)


@functools.lru_cache(maxsize=10)
def prun_data(model_name=None, pruning_method=None, task_dataset=None, max_rows=None) -> DataFrame:
    """
    Get pruning analytics as a pandas DataFrame.
    
    Extracts pruning experiment results from the 'prun' table, showing model compression
    metrics achieved through various pruning methods.

    Parameters:
      - model_name (str | None): filter by model name (FK to nn.name)
      - pruning_method (str | None): filter by pruning method (e.g., 'magnitude', 'lottery_ticket')
      - task_dataset (str | None): filter by task_dataset combination
      - max_rows (int | None): maximum number of results

    Returns:
      - A pandas DataFrame with columns:
        'id', 'model_name', 'pruning_method', 'task_dataset', 'status', 'accuracy',
        'duration', 'pruning_ratio', 'params_before', 'params_after', 'params_removed',
        'model_size_before_kb', 'model_size_after_kb'
        
    Each row represents one pruning experiment result with compression metrics.
    Useful for analyzing model compression effectiveness across different pruning techniques.
    """
    dt: tuple[dict, ...] = DB_Read.prun_data(
        model_name=model_name,
        pruning_method=pruning_method,
        task_dataset=task_dataset,
        max_rows=max_rows
    )
    return DataFrame.from_records(dt)


@functools.lru_cache(maxsize=10)
def nn_stat_data(nn_name=None, prm_id=None, max_rows=None) -> DataFrame:
    """
    Get NN architecture statistics as a pandas DataFrame.

    Parameters:
      - nn_name (str | None): filter by neural network name
      - prm_id (str | None): filter by parameter configuration ID
      - max_rows (int | None): maximum number of results

    Returns:
      - A pandas DataFrame with columns:
        'id', 'nn_name', 'prm_id',
        'total_layers', 'leaf_layers', 'max_depth',
        'total_params', 'trainable_params', 'frozen_params',
        'flops', 'model_size_mb', 'buffer_size_mb', 'total_memory_mb',
        'dropout_count', 'has_attention', 'has_residual_connections',
        'is_resnet_like', 'is_vgg_like', 'is_inception_like',
        'is_densenet_like', 'is_unet_like', 'is_transformer_like',
        'is_mobilenet_like', 'is_efficientnet_like',
        'code_length', 'num_classes_defined', 'num_functions_defined',
        'uses_sequential', 'uses_modulelist', 'uses_moduledict',
        'meta' (dict with additional metadata), 'error'
    """
    dt: tuple[dict, ...] = DB_Read.nn_stat_data(nn_name=nn_name, prm_id=prm_id, max_rows=max_rows)
    return DataFrame.from_records(dt)


def check_nn(nn_code: str, task: str, dataset: str, metric: str, prm: dict, save_to_db=True, prefix=None, save_path=None, export_onnx=False,
             epoch_limit_minutes=default_epoch_limit_minutes, transform_dir=None) -> tuple[str, float, float, float]:
    """
    Train the new NN model with the provided hyperparameters (prm) and save it to the database if training is successful.
    for argument description see :ref:`ab.nn.util.db.Write.save_nn()`
    :return: Automatically generated name of NN model, its accuracy, accuracy to time metric, and quality of the code metric.
    """
    return Train.train_new(nn_code, task, dataset, metric, prm, save_to_db=save_to_db, prefix=prefix, save_path=save_path, export_onnx=export_onnx,
                           epoch_limit_minutes=epoch_limit_minutes, transform_dir=transform_dir)
