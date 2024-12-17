"""
This API allows you to perform an objective assessment of data quality –
before proceeding with the exhaustive AI/ML exercise, it’s worthwhile to
check if your data is good enough. EazyML’s functions – Augmented Intelligence
and Overlap Factor – derive the metric from data to alert you of data
shortfalls for various measures – from data and model drift to completeness
and bias.
"""
from flask import Response

from .src.utils import (
                    quality_alert_helper,
                    transparency_api as tr_api,
                    utility
)

from .src.main import (
    ez_correlation_local,
    ez_data_balance_local,
    ez_impute_local,
    ez_outlier_local,
    ez_shape_local
)

import json
from functools import partial
convert_json = partial(json.dumps, indent=4, sort_keys=True, default=str)

def ez_data_quality(filename, outcome, options):
    """
    Performs a series of data quality checks on the given dataset and
    returns a JSON response indicating the results of these checks.

    Parameters:
        - **filename** (str):
            The path to the file containing the dataset.
        - **outcome** (str):
            The target variable (outcome) to assess data quality against.
        - **options** (dict, optional):
            A dictionary specifying additional configurations for data quality checks. 
        
    Returns :
        - **Dictionary with Fields** :
            - **success** (bool): Indicates whether the inference was successful.

    Example :
        .. code-block:: python

            ez_data_quality(
                filename = 'train/file/path.csv',
            )
    """
    if not filename or not outcome:
        return Response(response=convert_json(
            {
                "success": False,
                "message": tr_api.MANDATORY_PARAMETER % (["filename", "outcome"]),
            }
        ),
            status=400,
            mimetype="application/json",
        )

    if options:
        ez_config = options
        if not isinstance(ez_config, dict):
            return Response(response=convert_json(
                {
                    "success": False,
                    "message": tr_api.VALID_DATATYPE_DICT.replace("this", "options"),
                }
            ),
                status=422,
                mimetype="application/json",
            )
    else:
        ez_config = {}
    # Check for valid keys in the options dict
    is_string = lambda x: isinstance(x, str)
    for key in ez_config:
        if key not in tr_api.EZ_DATA_QUALITY_OPTIONS_KEYS_LIST:
            return Response(response=convert_json(
                {
                    "success": False,
                    "message": tr_api.INVALID_KEY % (key),
                }
            ),
                status=422,
                mimetype="application/json",
            )
        if "data_quality_options" == key:
            if type(ez_config[key]) != type({}):
                return Response(
                    response=convert_json(
                        {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % (key)}
                    ),
                    status=422,
                    mimetype="application/json",
                )
            continue
        if "prediction_filename" == key:
            continue
        if (not is_string(ez_config[key]) or not ez_config[key] in ["yes", "no"]):
            return Response(
                response=convert_json(
                    {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % (key)}
                ),
                status=422,
                mimetype="application/json",
            )

    if "data_quality_options" in ez_config:
        data_quality_options = ez_config["data_quality_options"]
    else:
        data_quality_options = {}

    for key in data_quality_options:
        if key not in tr_api.EZ_DATA_QUALITY_OPTIONS_OPTIONS_KEYS_LIST:
            return Response(response=convert_json(
                {
                    "success": False,
                    "message": tr_api.INVALID_KEY % (key),
                }
            ),
                status=422,
                mimetype="application/json",
            )
    if "impute" in ez_config and ez_config["impute"] == "yes":
        ez_load_options = {
            "outcome": outcome,
            "accelerate": "no",
            "impute": "no",
            "outlier": "no",
            "shuffle": "no"
        }
    else:
        ez_load_options = {
            "outcome": outcome,
            "accelerate": "yes",
            "impute": "no",
            "outlier": "no",
            "shuffle": "no"
        }
    if "data_load_options" in data_quality_options:
        tmp_options = data_quality_options["data_load_options"]
        for key in tmp_options:
            if key not in ["outcome"]:
                ez_load_options[key] = tmp_options[key]
    df = utility.get_df(filename)
    try:
        final_resp = {}

        if "data_shape" in ez_config and ez_config["data_shape"] == "yes":
            json_resp, status_code = ez_shape_local(df)
            # print("status code", status_code)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_shape_quality"] = json.loads(json_resp)
        # print('data_shape_quality', final_resp["data_shape_quality"])
        if "data_emptiness" in ez_config and ez_config["data_emptiness"] == "yes":
            impute_options = dict()
            if "impute" in ez_config:
                impute_options["impute"] = ez_config["impute"]
            # print('before impute')
            json_resp, status_code = ez_impute_local(df)
            # print('after impute', json_resp)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_emptiness_quality"] = json.loads(json_resp)
        if "remove_outliers" in ez_config and ez_config["remove_outliers"] == "yes":
            outlier_options = dict()
            if "remove_outliers" in ez_config:
                outlier_options["remove_outliers"] = ez_config["remove_outliers"]
            json_resp, status_code = ez_outlier_local(df)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_outliers_quality"] = json.loads(json_resp)
        # print('data_outliers_quality', final_resp["data_outliers_quality"])
        if "data_balance" in ez_config and ez_config["data_balance"] == "yes":
            json_resp, status_code = ez_data_balance_local(df, outcome)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_balance_quality"] = json.loads(json_resp)
        # print('data_balance_quality', final_resp["data_balance_quality"])
        if "outcome_correlation" in ez_config and ez_config["outcome_correlation"] == "yes":
            json_resp, status_code = ez_correlation_local(df, outcome)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_correlation_quality"] = json.loads(json_resp)

        alerts = quality_alert_helper.quality_alerts(final_resp)
        final_resp["data_bad_quality_alerts"] = alerts

        final_resp["success"] = True
        final_resp["message"] = tr_api.DATA_QUALITY_SUCCESS
        return  final_resp

    except Exception as e:
        return e