import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class IDFModelEvaluator:
    def __init__(self):
        self.models = {
            'Gumbel': 'results/idf_data.csv',
            'SVM': 'results/idf_curves_SVM.csv',
            'ANN': 'results/idf_curves_ANN.csv',
            'TCN': 'results/idf_curves_TCN.csv',
            'TCAN': 'results/idf_curves_TCAN.csv'
        }
        self.historical_data = None
        self.model_data = {}
        self.results = {}
        
    def load_data(self):
        """Load historical and model data"""
        # Load historical annual maximum intensity data
        self.historical_data = pd.read_csv('results/annual_max_intensity.csv')
        
        # Load model IDF curves
        for model_name, filepath in self.models.items():
            df = pd.read_csv(filepath)
            if model_name == 'Gumbel':
                # Gumbel data has different structure - transpose it
                df = df.set_index('Return Period (years)').T
                df.index = [5, 10, 15, 30, 60, 180, 1440]  # Duration in minutes
                df.index.name = 'Duration (minutes)'
            else:
                # Other models have duration as first column
                df = df.set_index('Duration (minutes)')
            self.model_data[model_name] = df
        
        print("Data loaded successfully!")
        print(f"Historical data shape: {self.historical_data.shape}")
        for model, data in self.model_data.items():
            print(f"{model} data shape: {data.shape}")
    
    def calculate_historical_statistics(self):
        """Calculate statistical measures from historical data"""
        durations = ['5mns', '10mns', '15mns', '30mns', '1h', '3h', '24h']
        duration_minutes = [5, 10, 15, 30, 60, 180, 1440]
        
        historical_stats = {}
        
        for i, duration in enumerate(durations):
            if duration in self.historical_data.columns:
                values = self.historical_data[duration].values
                historical_stats[duration_minutes[i]] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'percentiles': {
                        '50': np.percentile(values, 50),
                        '80': np.percentile(values, 80),
                        '90': np.percentile(values, 90),
                        '95': np.percentile(values, 95),
                        '99': np.percentile(values, 99)
                    }
                }
        
        return historical_stats
    
    def estimate_return_periods(self, historical_stats):
        """Estimate return periods for historical data using empirical approach"""
        return_period_estimates = {}
        
        for duration_min, data_stats in historical_stats.items():
            # Use Weibull plotting position formula: T = (n+1)/(m)
            # where n = number of years, m = rank
            
            # Approximate return periods based on percentiles
            return_period_estimates[duration_min] = {
                2: data_stats['percentiles']['50'],    # 2-year ≈ 50th percentile
                5: data_stats['percentiles']['80'],    # 5-year ≈ 80th percentile
                10: data_stats['percentiles']['90'],   # 10-year ≈ 90th percentile
                25: data_stats['percentiles']['95'],   # 25-year ≈ 95th percentile
                50: data_stats['percentiles']['99'],   # 50-year ≈ 99th percentile
                100: data_stats['max']                 # 100-year ≈ maximum observed
            }
        
        return return_period_estimates
    
    def compare_models(self):
        """Compare all models against historical data"""
        historical_stats = self.calculate_historical_statistics()
        historical_return_periods = self.estimate_return_periods(historical_stats)
        
        comparison_results = {}
        
        for model_name, model_data in self.model_data.items():
            model_results = {
                'rmse': {},
                'mae': {},
                'r2': {},
                'bias': {},
                'overall_score': 0
            }
            
            total_rmse = 0
            total_mae = 0
            total_r2 = 0
            total_bias = 0
            n_comparisons = 0
            
            # Compare for each duration
            for duration_min in historical_return_periods.keys():
                if duration_min in model_data.index:
                    # Get historical and model values for different return periods
                    historical_values = []
                    model_values = []
                    
                    for return_period in [2, 5, 10, 25, 50, 100]:
                        if return_period in historical_return_periods[duration_min]:
                            historical_values.append(historical_return_periods[duration_min][return_period])
                            
                            # Get model prediction for this return period
                            if f'{return_period}-year' in model_data.columns:
                                model_values.append(model_data.loc[duration_min, f'{return_period}-year'])
                            elif return_period in model_data.columns:
                                model_values.append(model_data.loc[duration_min, return_period])
                    
                    if len(historical_values) > 0 and len(model_values) > 0:
                        historical_values = np.array(historical_values)
                        model_values = np.array(model_values)
                        
                        # Calculate metrics
                        rmse = np.sqrt(mean_squared_error(historical_values, model_values))
                        mae = mean_absolute_error(historical_values, model_values)
                        r2 = r2_score(historical_values, model_values)
                        bias = np.mean(model_values - historical_values)
                        
                        model_results['rmse'][duration_min] = rmse
                        model_results['mae'][duration_min] = mae
                        model_results['r2'][duration_min] = r2
                        model_results['bias'][duration_min] = bias
                        
                        total_rmse += rmse
                        total_mae += mae
                        total_r2 += r2
                        total_bias += abs(bias)
                        n_comparisons += 1
            
            # Calculate overall performance score
            if n_comparisons > 0:
                avg_rmse = total_rmse / n_comparisons
                avg_mae = total_mae / n_comparisons
                avg_r2 = total_r2 / n_comparisons
                avg_bias = total_bias / n_comparisons
                
                # Overall score (lower is better for RMSE, MAE, bias; higher is better for R2)
                model_results['overall_score'] = {
                    'rmse': avg_rmse,
                    'mae': avg_mae,
                    'r2': avg_r2,
                    'bias': avg_bias,
                    'composite_score': (1 - avg_r2) + (avg_rmse/100) + (avg_mae/100) + (avg_bias/100)
                }
            
            comparison_results[model_name] = model_results
        
        return comparison_results, historical_stats, historical_return_periods
    
    def create_visualizations(self, comparison_results, historical_stats, historical_return_periods):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('IDF Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Overall performance comparison
        ax1 = axes[0, 0]
        models = list(comparison_results.keys())
        composite_scores = [comparison_results[model]['overall_score']['composite_score'] 
                          for model in models if comparison_results[model]['overall_score']]
        
        bars = ax1.bar(models, composite_scores, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax1.set_title('Overall Model Performance\n(Lower is Better)', fontweight='bold')
        ax1.set_ylabel('Composite Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, composite_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. R² comparison
        ax2 = axes[0, 1]
        r2_scores = [comparison_results[model]['overall_score']['r2'] 
                    for model in models if comparison_results[model]['overall_score']]
        
        bars = ax2.bar(models, r2_scores, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax2.set_title('R² Scores\n(Higher is Better)', fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. RMSE comparison
        ax3 = axes[0, 2]
        rmse_scores = [comparison_results[model]['overall_score']['rmse'] 
                      for model in models if comparison_results[model]['overall_score']]
        
        bars = ax3.bar(models, rmse_scores, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax3.set_title('RMSE Values\n(Lower is Better)', fontweight='bold')
        ax3.set_ylabel('RMSE')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, rmse_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 4. Historical data distribution
        ax4 = axes[1, 0]
        durations = ['5mns', '10mns', '15mns', '30mns', '1h', '3h', '24h']
        for duration in durations:
            if duration in self.historical_data.columns:
                ax4.hist(self.historical_data[duration], alpha=0.6, label=duration, bins=15)
        ax4.set_title('Historical Data Distribution', fontweight='bold')
        ax4.set_xlabel('Intensity (mm/h)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5. Model predictions vs historical estimates
        ax5 = axes[1, 1]
        duration_min = 60  # 1 hour
        if duration_min in historical_return_periods:
            return_periods = [2, 5, 10, 25, 50, 100]
            historical_vals = [historical_return_periods[duration_min].get(rp, np.nan) for rp in return_periods]
            
            for model_name, model_data in self.model_data.items():
                if duration_min in model_data.index:
                    model_vals = []
                    for rp in return_periods:
                        if f'{rp}-year' in model_data.columns:
                            model_vals.append(model_data.loc[duration_min, f'{rp}-year'])
                        elif rp in model_data.columns:
                            model_vals.append(model_data.loc[duration_min, rp])
                        else:
                            model_vals.append(np.nan)
                    
                    ax5.plot(return_periods, model_vals, marker='o', label=model_name, linewidth=2)
            
            ax5.plot(return_periods, historical_vals, marker='s', label='Historical', 
                    linewidth=3, color='black', markersize=8)
            ax5.set_title('1-Hour Duration Comparison', fontweight='bold')
            ax5.set_xlabel('Return Period (years)')
            ax5.set_ylabel('Intensity (mm/h)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Bias analysis
        ax6 = axes[1, 2]
        bias_scores = [comparison_results[model]['overall_score']['bias'] 
                      for model in models if comparison_results[model]['overall_score']]
        
        colors = ['red' if bias > 0 else 'blue' for bias in bias_scores]
        bars = ax6.bar(models, bias_scores, color=colors)
        ax6.set_title('Model Bias\n(Positive=Overestimate, Negative=Underestimate)', fontweight='bold')
        ax6.set_ylabel('Bias (mm/h)')
        ax6.tick_params(axis='x', rotation=45)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, score in zip(bars, bias_scores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('figures/model_evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, comparison_results, historical_stats):
        """Generate detailed evaluation report"""
        print("="*80)
        print("COMPREHENSIVE IDF MODEL EVALUATION REPORT")
        print("="*80)
        
        print("\n1. HISTORICAL DATA SUMMARY")
        print("-" * 40)
        for duration_min, data_stats in historical_stats.items():
            duration_label = f"{duration_min} minutes"
            if duration_min == 60:
                duration_label = "1 hour"
            elif duration_min == 180:
                duration_label = "3 hours"
            elif duration_min == 1440:
                duration_label = "24 hours"
            
            print(f"\n{duration_label}:")
            print(f"  Mean: {data_stats['mean']:.2f} mm/h")
            print(f"  Std Dev: {data_stats['std']:.2f} mm/h")
            print(f"  Range: {data_stats['min']:.2f} - {data_stats['max']:.2f} mm/h")
        
        print("\n\n2. MODEL PERFORMANCE RANKING")
        print("-" * 40)
        
        # Sort models by composite score (lower is better)
        model_scores = [(model, results['overall_score']['composite_score']) 
                       for model, results in comparison_results.items() 
                       if results['overall_score']]
        model_scores.sort(key=lambda x: x[1])
        
        for i, (model, score) in enumerate(model_scores):
            print(f"{i+1}. {model:10} - Composite Score: {score:.4f}")
        
        print("\n\n3. DETAILED PERFORMANCE METRICS")
        print("-" * 40)
        
        for model, results in comparison_results.items():
            if results['overall_score']:
                print(f"\n{model.upper()}:")
                print(f"  R² Score:    {results['overall_score']['r2']:.4f}")
                print(f"  RMSE:        {results['overall_score']['rmse']:.2f} mm/h")
                print(f"  MAE:         {results['overall_score']['mae']:.2f} mm/h")
                print(f"  Bias:        {results['overall_score']['bias']:.2f} mm/h")
                print(f"  Composite:   {results['overall_score']['composite_score']:.4f}")
        
        print("\n\n4. RECOMMENDATIONS")
        print("-" * 40)
        
        best_model = model_scores[0][0]
        best_score = model_scores[0][1]
        
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
        print(f"   Composite Score: {best_score:.4f}")
        
        best_results = comparison_results[best_model]['overall_score']
        print("\n   Key Strengths:")
        print(f"   • R² Score: {best_results['r2']:.4f} (explains {best_results['r2']*100:.1f}% of variance)")
        print(f"   • RMSE: {best_results['rmse']:.2f} mm/h")
        print(f"   • Bias: {best_results['bias']:.2f} mm/h ({'overestimates' if best_results['bias'] > 0 else 'underestimates'})")
        
        print("\n   Reliability Assessment:")
        if best_results['r2'] > 0.8:
            print("   ✓ Excellent correlation with historical data")
        elif best_results['r2'] > 0.6:
            print("   ✓ Good correlation with historical data")
        else:
            print("   ⚠ Moderate correlation with historical data")
        
        if abs(best_results['bias']) < 5:
            print("   ✓ Low bias - well-calibrated predictions")
        elif abs(best_results['bias']) < 10:
            print("   ⚠ Moderate bias - may need calibration")
        else:
            print("   ⚠ High bias - requires significant calibration")
        
        print("\n   Recommended Use Cases:")
        print("   • Design storm estimation")
        print("   • Flood risk assessment")
        print("   • Infrastructure planning")
        
        # Compare with traditional method
        if 'Gumbel' in comparison_results:
            gumbel_score = comparison_results['Gumbel']['overall_score']['composite_score']
            improvement = ((gumbel_score - best_score) / gumbel_score) * 100
            print(f"\n   Improvement over Gumbel: {improvement:.1f}%")
        
        print("\n" + "="*80)
        
        return best_model, best_results
    
    def run_evaluation(self):
        """Run complete evaluation process"""
        print("Starting comprehensive IDF model evaluation...")
        
        # Load data
        self.load_data()
        
        # Perform comparison
        comparison_results, historical_stats, historical_return_periods = self.compare_models()
        
        # Create visualizations
        self.create_visualizations(comparison_results, historical_stats, historical_return_periods)
        
        # Generate report
        best_model, best_results = self.generate_detailed_report(comparison_results, historical_stats)
        
        # Save results
        results_df = pd.DataFrame({
            model: {
                'R2': results['overall_score']['r2'] if results['overall_score'] else np.nan,
                'RMSE': results['overall_score']['rmse'] if results['overall_score'] else np.nan,
                'MAE': results['overall_score']['mae'] if results['overall_score'] else np.nan,
                'Bias': results['overall_score']['bias'] if results['overall_score'] else np.nan,
                'Composite_Score': results['overall_score']['composite_score'] if results['overall_score'] else np.nan
            }
            for model, results in comparison_results.items()
        }).T
        
        results_df.to_csv('results/model_evaluation_results.csv')
        
        print("\n✅ Evaluation complete! Results saved to 'results/model_evaluation_results.csv'")
        print("📊 Visualizations saved to 'figures/model_evaluation_comprehensive.png'")
        
        return best_model, comparison_results

# Run the evaluation
if __name__ == "__main__":
    evaluator = IDFModelEvaluator()
    best_model, results = evaluator.run_evaluation()