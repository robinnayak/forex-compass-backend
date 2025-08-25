from rest_framework import serializers
import numpy as np
import pandas as pd

class AnalysisResultSerializer(serializers.Serializer):
    rsi = serializers.ListField(child=serializers.FloatField(allow_null=True), allow_null=True)
    signals = serializers.DictField(child=serializers.BooleanField(allow_null=True))
    analysis = serializers.DictField()

    def to_representation(self, instance):
        # Recursively replace NaN, inf, -inf with 0 or None for JSON compatibility
        def clean_value(val):
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return 0.0
            if isinstance(val, (np.floating, np.integer)) and (np.isnan(val) or np.isinf(val)):
                return 0.0
            if isinstance(val, pd.Series):
                # Convert boolean Series to list of bool, others to list of float
                if val.dtype == bool:
                    return [bool(x) for x in val.values]
                else:
                    return [float(x) if not (np.isnan(x) or np.isinf(x)) else 0.0 for x in val.values]
            if isinstance(val, pd.DataFrame):
                # Convert DataFrame to dict of lists, cleaning each value
                return {str(col): clean_value(val[col]) for col in val.columns}
            if isinstance(val, dict):
                # Ensure keys are strings and values are cleaned
                return {str(k): clean_value(v) for k, v in val.items()}
            if isinstance(val, list):
                return [clean_value(x) for x in val]
            return val
        rep = super().to_representation(instance)
        return clean_value(rep)
