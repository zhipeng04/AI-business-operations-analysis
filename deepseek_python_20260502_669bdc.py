"""
AI数据分析与经营复盘Agent
功能：自动分析经营数据，生成可视化报告和智能建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BusinessAnalysisAgent:
    """经营数据分析与复盘Agent"""
    
    def __init__(self, data_path=None):
        """
        初始化Agent
        :param data_path: 数据文件路径（CSV格式）
        """
        self.data = None
        self.analysis_results = {}
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """加载数据"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"✅ 成功加载数据，共 {len(self.data)} 条记录")
            print(f"📊 字段：{list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"❌ 加载数据失败：{e}")
            return False
    
    def generate_sample_data(self):
        """生成示例经营数据"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        data = {
            '日期': dates,
            '销售额': np.random.normal(10000, 2000, len(dates)),
            '成本': np.random.normal(6000, 1000, len(dates)),
            '客户数': np.random.poisson(50, len(dates)),
            '新产品销售': np.random.normal(3000, 800, len(dates)),
            '退货率': np.random.uniform(0.02, 0.08, len(dates)),
            '营销投入': np.random.normal(2000, 500, len(dates)),
            '员工数': np.random.randint(20, 30, len(dates)),
            '区域': np.random.choice(['华东', '华南', '华北', '西南'], len(dates))
        }
        
        self.data = pd.DataFrame(data)
        self.data['利润'] = self.data['销售额'] - self.data['成本']
        self.data['利润率'] = self.data['利润'] / self.data['销售额']
        self.data['客单价'] = self.data['销售额'] / self.data['客户数']
        self.data['月份'] = self.data['日期'].dt.month
        
        print("✅ 生成示例数据成功")
        return self.data
    
    def data_quality_check(self):
        """数据质量检查"""
        print("\n" + "="*50)
        print("📋 数据质量检查报告")
        print("="*50)
        
        quality_report = {
            '总记录数': len(self.data),
            '缺失值统计': self.data.isnull().sum().to_dict(),
            '重复记录数': self.data.duplicated().sum(),
            '数据类型': self.data.dtypes.to_dict()
        }
        
        for key, value in quality_report.items():
            print(f"{key}：{value}")
        
        return quality_report
    
    def basic_statistics(self):
        """基础统计分析"""
        print("\n" + "="*50)
        print("📈 基础统计分析")
        print("="*50)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats = self.data[numeric_cols].describe()
        print(stats.round(2))
        
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def trend_analysis(self):
        """趋势分析"""
        print("\n" + "="*50)
        print("📊 趋势分析")
        print("="*50)
        
        # 按月聚合
        self.data['月份'] = pd.to_datetime(self.data['日期']).dt.to_period('M')
        monthly = self.data.groupby('月份').agg({
            '销售额': 'sum',
            '利润': 'sum',
            '客户数': 'sum',
            '利润率': 'mean',
            '客单价': 'mean'
        }).reset_index()
        
        # 计算环比增长率
        monthly['销售额环比'] = monthly['销售额'].pct_change() * 100
        monthly['利润环比'] = monthly['利润'].pct_change() * 100
        
        print("\n月度经营趋势：")
        print(monthly.round(2))
        
        self.analysis_results['monthly_trend'] = monthly
        return monthly
    
    def anomaly_detection(self):
        """异常检测"""
        print("\n" + "="*50)
        print("🔍 异常检测")
        print("="*50)
        
        anomalies = {}
        
        # 使用Z-score检测异常值
        for col in ['销售额', '利润', '客户数']:
            if col in self.data.columns:
                mean = self.data[col].mean()
                std = self.data[col].std()
                z_scores = np.abs((self.data[col] - mean) / std)
                anomaly_idx = z_scores > 2
                
                if anomaly_idx.sum() > 0:
                    anomalies[col] = {
                        '异常记录数': anomaly_idx.sum(),
                        '异常记录': self.data[anomaly_idx][['日期', col]].to_dict('records')
                    }
                    print(f"\n{col} - 发现 {anomaly_idx.sum()} 条异常记录")
                    print(self.data[anomaly_idx][['日期', col]].to_string())
        
        self.analysis_results['anomalies'] = anomalies
        return anomalies
    
    def profitability_analysis(self):
        """盈利能力分析"""
        print("\n" + "="*50)
        print("💰 盈利能力分析")
        print("="*50)
        
        analysis = {
            '平均利润率': f"{self.data['利润率'].mean()*100:.2f}%",
            '平均客单价': f"¥{self.data['客单价'].mean():.2f}",
            '平均退货率': f"{self.data['退货率'].mean()*100:.2f}%",
            '营销投入占比': f"{(self.data['营销投入'].sum() / self.data['销售额'].sum())*100:.2f}%",
        }
        
        for key, value in analysis.items():
            print(f"{key}：{value}")
        
        self.analysis_results['profitability'] = analysis
        return analysis
    
    def generate_visualizations(self):
        """生成可视化图表"""
        fig = plt.figure(figsize=(15, 12))
        
        # 1. 销售趋势图
        ax1 = plt.subplot(2, 3, 1)
        monthly = self.data.groupby('月份').agg({'销售额': 'sum'}).reset_index()
        monthly['月份'] = monthly['月份'].astype(str)
        ax1.plot(monthly['月份'], monthly['销售额'], marker='o', linewidth=2, markersize=6)
        ax1.set_title('月度销售趋势', fontsize=14, fontweight='bold')
        ax1.set_xlabel('月份')
        ax1.set_ylabel('销售额（元）')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 利润分析
        ax2 = plt.subplot(2, 3, 2)
        profit_monthly = self.data.groupby('月份').agg({'利润': 'sum'}).reset_index()
        profit_monthly['月份'] = profit_monthly['月份'].astype(str)
        colors_profit = ['#ff6b6b' if x < 0 else '#51cf66' for x in profit_monthly['利润']]
        ax2.bar(profit_monthly['月份'], profit_monthly['利润'], color=colors_profit, edgecolor='white')
        ax2.set_title('月度利润分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('月份')
        ax2.set_ylabel('利润（元）')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 成本结构分析
        ax3 = plt.subplot(2, 3, 3)
        cost_components = {
            '产品成本': self.data['成本'].sum(),
            '营销成本': self.data['营销投入'].sum(),
        }
        wedges, texts, autotexts = ax3.pie(
            cost_components.values(), 
            labels=cost_components.keys(), 
            autopct='%1.1f%%',
            colors=['#ffd43b', '#74c0fc'],
            startangle=90
        )
        ax3.set_title('成本结构分析', fontsize=14, fontweight='bold')
        
        # 4. 利润率分布
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(self.data['利润率'], bins=30, color='#748ffc', edgecolor='white', alpha=0.7)
        ax4.axvline(self.data['利润率'].mean(), color='red', linestyle='--', linewidth=2, label=f'平均利润率：{self.data["利润率"].mean()*100:.2f}%')
        ax4.set_title('利润率分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('利润率')
        ax4.set_ylabel('频数')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 区域销售对比
        ax5 = plt.subplot(2, 3, 5)
        region_sales = self.data.groupby('区域')['销售额'].sum().sort_values(ascending=True)
        ax5.barh(region_sales.index, region_sales.values, color='#69db7c', edgecolor='white')
        ax5.set_title('区域销售对比', fontsize=14, fontweight='bold')
        ax5.set_xlabel('销售额（元）')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. 关键指标趋势
        ax6 = plt.subplot(2, 3, 6)
        metrics_monthly = self.data.groupby('月份').agg({
            '客单价': 'mean',
            '退货率': 'mean',
            '利润率': 'mean'
        }).reset_index()
        metrics_monthly['月份'] = metrics_monthly['月份'].astype(str)
        
        ax6_twin = ax6.twinx()
        line1, = ax6.plot(metrics_monthly['月份'], metrics_monthly['客单价'], 
                         marker='s', color='#4c6ef5', linewidth=2, label='客单价')
        line2, = ax6_twin.plot(metrics_monthly['月份'], metrics_monthly['利润率']*100, 
                              marker='^', color='#f03e3e', linewidth=2, label='利润率(%)')
        
        ax6.set_title('关键指标趋势', fontsize=14, fontweight='bold')
        ax6.set_xlabel('月份')
        ax6.set_ylabel('客单价（元）', color='#4c6ef5')
        ax6_twin.set_ylabel('利润率(%)', color='#f03e3e')
        ax6.tick_params(axis='x', rotation=45)
        
        # 合并图例
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax6.legend(lines, labels, loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('经营分析报告.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 可视化图表已生成并保存为'经营分析报告.png'")
    
    def generate_recommendations(self):
        """生成智能建议"""
        print("\n" + "="*50)
        print("💡 智能决策建议")
        print("="*50)
        
        recommendations = []
        
        # 基于数据的智能建议
        avg_profit_rate = self.data['利润率'].mean()
        avg_return_rate = self.data['退货率'].mean()
        sales_trend = self.data.groupby('月份')['销售额'].sum().pct_change().mean()
        
        if avg_profit_rate < 0.15:
            recommendations.append({
                'priority': '高',
                'category': '盈利能力',
                'suggestion': f'当前平均利润率仅为{avg_profit_rate*100:.1f}%，低于健康水平。建议：\n'
                             f'  1. 审查产品定价策略\n'
                             f'  2. 优化供应链降低采购成本\n'
                             f'  3. 提升高毛利产品销售占比'
            })
        
        if avg_return_rate > 0.05:
            recommendations.append({
                'priority': '中',
                'category': '质量控制',
                'suggestion': f'退货率达到{avg_return_rate*100:.1f}%，需要关注产品质量。建议：\n'
                             f'  1. 加强质量检测流程\n'
                             f'  2. 分析退货原因分布\n'
                             f'  3. 优化产品描述准确性'
            })
        
        if sales_trend < 0:
            recommendations.append({
                'priority': '高',
                'category': '销售增长',
                'suggestion': f'销售增长呈现下行趋势，环比变化{sales_trend*100:.1f}%。建议：\n'
                             f'  1. 加大营销推广力度\n'
                             f'  2. 开发新客户渠道\n'
                             f'  3. 推出促销活动刺激消费'
            })
        
        recommendations.append({
            'priority': '低',
            'category': '数据驱动',
            'suggestion': f'建议建立数据驱动的决策机制：\n'
                         f'  1. 每日监控关键经营指标\n'
                         f'  2. 建立预警机制（销售额/利润率异常波动）\n'
                         f'  3. 定期进行经营复盘会议\n'
                         f'  4. 使用AI预测模型进行趋势预判'
        })
        
        # 输出建议
        for i, rec in enumerate(recommendations, 1):
            print(f"\n建议 {i}：")
            print(f"优先级：{rec['priority']} | 类别：{rec['category']}")
            print(f"内容：{rec['suggestion']}")
            print("-" * 50)
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def generate_report(self, output_path='经营复盘报告.txt'):
        """生成完整的文本报告"""
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("📊 经营数据分析与复盘报告")
        report_lines.append("="*60)
        report_lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("-"*60)
        
        # 执行所有分析
        self.data_quality_check()
        stats = self.basic_statistics()
        monthly = self.trend_analysis()
        anomalies = self.anomaly_detection()
        profitability = self.profitability_analysis()
        recommendations = self.generate_recommendations()
        
        # 生成可视化
        self.generate_visualizations()
        
        print(f"\n\n✅ 完整报告已生成！")
        print(f"📄 可视化图表：经营分析报告.png")
        print(f"📝 数据摘要已准备好")
        
        return self.analysis_results
    
    def save_report_to_file(self, filename='经营复盘报告.md'):
        """保存报告到Markdown文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# 📊 经营数据分析与复盘报告\n\n")
            f.write(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'profitability' in self.analysis_results:
                f.write("## 💰 盈利能力概览\n\n")
                for key, value in self.analysis_results['profitability'].items():
                    f.write(f"- **{key}：** {value}\n")
                f.write("\n")
            
            if 'recommendations' in self.analysis_results:
                f.write("## 💡 智能建议\n\n")
                for i, rec in enumerate(self.analysis_results['recommendations'], 1):
                    f.write(f"### 建议 {i}：{rec['category']}（优先级：{rec['priority']}）\n\n")
                    f.write(f"{rec['suggestion']}\n\n")
            
            if 'anomalies' in self.analysis_results:
                f.write("## 🔍 异常检测\n\n")
                for metric, details in self.analysis_results['anomalies'].items():
                    f.write(f"- **{metric}：** 发现 {details['异常记录数']} 条异常\n")
        
        print(f"✅ Markdown报告已保存至：{filename}")


# ==================== 主程序 ====================
def main():
    """主函数：运行经营分析Agent"""
    print("🚀 启动AI经营分析Agent...\n")
    
    # 创建Agent实例
    agent = BusinessAnalysisAgent()
    
    # 生成示例数据（实际使用时可以替换为agent.load_data('your_data.csv')）
    print("📦 生成示例经营数据...")
    agent.generate_sample_data()
    
    # 执行完整分析
    print("\n" + "🔄 开始执行分析...".center(60, "="))
    analysis_results = agent.generate_report()
    
    # 保存报告到文件
    agent.save_report_to_file()
    
    print("\n" + "="*60)
    print("✨ 分析完成！✨".center(60))
    print("="*60)
    print("\n📌 提示：")
    print("1. 查看生成的'经营分析报告.png'获取可视化图表")
    print("2. 查看'经营复盘报告.md'获取详细文本报告")
    print("3. 如需分析真实数据，使用 agent.load_data('your_file.csv')")


if __name__ == "__main__":
    main()