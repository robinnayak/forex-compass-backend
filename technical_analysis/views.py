# In your views.py
from technical_analysis.data_quality.processor import DataQualityProcessor
from technical_analysis.utils import calculate_all_indicators
import pandas as pd
from django.http import JsonResponse
import json
import numpy as np
from rest_framework.views import APIView
from technical_analysis.serializers import AnalysisResultSerializer
from rest_framework.response import Response    

def welcome_screen(request):
    return JsonResponse({"message": "Welcome to the Technical Analysis API"})


class AnalyzeMarketData(APIView):
    def get(self, request):
        data = pd.read_csv('technical_analysis/data/EURUSD1.csv')
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data.set_index('Date', inplace=True)
        
        print(f"Data testing: {data.shape}")
        print(f"Date type: {data.index.dtype}")
        data = DataQualityProcessor.clean_ohlcv_data(data)
        print(f"Data testing after: {data.shape}")

        try:
            # Safety check: ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                print(f"Data cleaning returned type: {type(data)}")
                return Response({"error": "Data cleaning did not return a DataFrame."}, status=500)
            if not data.empty:
                analysis_results = calculate_all_indicators(data, time_frame="1min")
                print("analysis result:", analysis_results)
                # Use serializer to clean output for JSON
                serializer = AnalysisResultSerializer(analysis_results)
                return Response(serializer.data, status=200)
            else:
                print(data.columns)
                return Response({"message": "Data processed successfully, but no data rows remain."}, status=200)
        except Exception as e:
            print(e)
            return Response({"error": str(e)}, status=500)
