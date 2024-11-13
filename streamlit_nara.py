import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import requests
import seaborn as sns
from io import BytesIO
import xlsxwriter
from scipy.stats import norm

# 발주처 카테고리 분류 함수
def categorize_client(client):
    """발주처를 카테고리로 분류하는 함수"""
    if pd.isna(client):
        return '기타'
        
    client = str(client).strip()
    
    # 군사기관 (가장 먼저 검사)
    if any(word in client for word in [
        '군', '공군', '해군', '육군', '국방', '국군', '군사', '군단', 
        '사령부', '군수', '군비', '군시설', '군사시설', '군부대'
    ]):
        return '군사기관'
    
    # 서울특별시교육청 관련 기관
    if '서울특별시교육청' in client:
        return '서울특별시교육청'
    
    # 서울특별시 관련 기관
    if '서울특별시' in client or '서울시' in client:
        return '서울특별시'
    
    # 교육기관 분류
    if any(word in client for word in ['교육청', '교육지원청']):
        return '교육청'
    elif any(word in client for word in ['고등학교', '중학교', '초등학교', '학교']):
        return '학교'
    elif '대학교' in client:
        return '대학교'
    
    # 중앙행정기관 세분화
    if '국토교통부' in client or '국토부' in client:
        return '국토교통부'
    elif '조달청' in client:
        return '조달청'
    elif any(org in client for org in ['청', '부', '처', '원']) and not any(word in client for word in ['학교', '지원청']):
        return '중앙행정기관'
    
    # 지방자치단체
    if any(word in client for word in ['시청', '도청', '군청', '구청', '시장', '도지사', '군수', '구청장']):
        return '지방자치단체'
    
    # 기타공사로 분류될 기관들
    if any(org in client for org in [
        '한국전력공사', '한국도로공사', '한국토지주택공사', '한국수자원공사',
        '한국철도공사', '한국농어촌공사', '대한석탄공사', '한국마사회'
    ]):
        return '기타공사'
    
    # 기타공단으로 분류될 기관들
    if any(org in client for org in [
        '국민건강보험공단', '한국환경공단', '한국산업단지공단',
        '근로복지공단', '한국가스공단', '도로교통공단', '국민연금공단'
    ]):
        return '기타공단'
    
    # 나머지는 기타로 분류
    return '기타'

class BidPricePredictor:
    def __init__(self, data):
        # 결측치가 없는 데이터만 선택
        required_columns = ['기초금액', '추정가격', '투찰률', 'A값', '순공사원가', '발주처_카테고리', '1순위사정률']
        self.data = data.dropna(subset=required_columns).copy()
        self.model = None
        self.scaler = StandardScaler()
        self.category_encoder = None  # 카테고리 인코딩을 위한 변수 추가
        
        st.text("=== 모델 초기화 ===")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("전체 데이터 수", f"{len(data):,}개")
        with col2:
            st.metric("학습 가능한 데이터 수", f"{len(self.data):,}개")
    
    def train_model(self):
        try:
            # 수치형 특성과 범주형 특성 분리
            numeric_features = ['기초금액', '추정가격', '투찰률', 'A값', '순공사원가']
            categorical_features = ['발주처_카테고리']
            target = '1순위사정률'
            
            # 수치형 특성 스케일링
            X_numeric = self.data[numeric_features]
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            
            # 범주형 특성 원-핫 인코딩
            X_categorical = pd.get_dummies(self.data[categorical_features], prefix=categorical_features)
            self.category_encoder = {col: idx for idx, col in enumerate(X_categorical.columns)}
            
            # 특성 결합
            X = np.hstack([X_numeric_scaled, X_categorical])
            y = self.data[target]
            
            # 학습/테스트 데이터 분할
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 모델 학습
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(self.X_train, self.y_train)
            
            # 특성 중요도 시각화
            st.subheader("특성 중요도")
            feature_names = numeric_features + list(X_categorical.columns)
            feature_importance_df = pd.DataFrame({
                '특성': feature_names,
                '중요도': self.model.feature_importances_
            })
            
            fig = plot_feature_importance(self.model, feature_importance_df)
            if fig is not None:
                st.pyplot(fig)
            
            return self.model, self.data
            
        except Exception as e:
            st.error(f"모델 학습 중 오류가 발생했습니다: {str(e)}")
            return None, None
    
    def predict_rate(self, new_data):
        if self.model is None:
            st.error("모델이 학습되지 않았습니다. 먼저 train_model()을 실행하세요.")
            return None
        
        try:
            # 데이터프레임으로 변환
            if isinstance(new_data, dict):
                new_data = pd.DataFrame([new_data])
            
            # 발주처 카테고리 추가
            if '발주처_카테고리' not in new_data.columns and '발주처' in new_data.columns:
                new_data['발주처_카테고리'] = new_data['발주처'].apply(categorize_client)
            
            # 수치형 특성 스케일링
            numeric_features = ['기초금액', '추정가격', '투찰률', 'A값', '순공사원가']
            X_numeric = new_data[numeric_features]
            X_numeric_scaled = self.scaler.transform(X_numeric)
            
            # 범주형 특성 원-핫 인코딩
            X_categorical = pd.get_dummies(new_data[['발주처_카테고리']], prefix=['발주처_카테고리'])
            
            # 누락된 카테고리 열 추가
            for col in self.category_encoder.keys():
                if col not in X_categorical.columns:
                    X_categorical[col] = 0
            
            # 학습 시 사용된 열 순서대로 정렬
            X_categorical = X_categorical[list(self.category_encoder.keys())]
            
            # 특성 결합
            X = np.hstack([X_numeric_scaled, X_categorical])
            
            # 예측 수행
            predicted = self.model.predict(X)
            
            return predicted[0] if len(predicted) == 1 else predicted
            
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
            return None



def show_category_statistics(data):
    """발주처 카테고리별 통계를 계산하고 그래프로 표시하는 함수"""
    try:
        # 1순위사정률이 있는 데이터만 선택
        valid_data = data.dropna(subset=['1순위사정률'])
        
        # 카테고리별 통계 계산
        stats = valid_data.groupby('발주처_카테고리').agg({
            '1순위사정률': [
                ('건수', 'count'),
                ('평균 사정률', 'mean'),
                ('최저 사정률', 'min'),
                ('최고 사정률', 'max'),
                ('중앙값 사정률', 'median'),
                ('표준편차', 'std')
            ]
        })
        
        # 멀티인덱스 처리
        stats.columns = stats.columns.droplevel()
        
        # 건수로 정렬
        stats = stats.sort_values('건수', ascending=False)
        
        # 정규분포 분석을 위한 함수
        def find_top_probabilities(group_data, mean, std):
            try:
                # 0.01% 단위로 구간 생성
                intervals = np.arange(97, 103, 0.01)
                
                # 99.5% ~ 100.5% 구간의 인덱스 찾기
                exclude_start = np.searchsorted(intervals, 99.5)
                exclude_end = np.searchsorted(intervals, 100.5)
                
                # 각 구간의 실제 발생 확률 계산
                actual_prob = np.array([
                    len(group_data[(group_data >= i) & (group_data < i + 0.01)]) / len(group_data)
                    for i in intervals
                ])
                
                # 정규분포 확률 계산
                normal_prob = norm.pdf(intervals, mean, std)
                normal_prob = normal_prob / sum(normal_prob)  # 정규화
                
                # 실제 확률이 정규분포 확률보다 높은 구간 찾기
                prob_diff = actual_prob - normal_prob
                
                # 제외 구간의 확률 차이를 -inf로 설정하여 선택되지 않도록 함
                prob_diff[exclude_start:exclude_end] = float('-inf')
                
                # 상위 5개 구간 찾기
                top_indices = np.argsort(prob_diff)[-5:][::-1]
                
                # 상위 5개 구간과 확률 차이 반환
                top_intervals = [f"{intervals[i]:.2f}" for i in top_indices]
                prob_differences = [f"{prob_diff[i]*100:.2f}" for i in top_indices]
                
                return top_intervals, prob_differences
            except Exception as e:
                st.error(f"정규분포 분석 중 오류 발생: {str(e)}")
                return [], []
        
        # 각 카테고리별로 정규분포 분석 수행
        for category in stats.index:
            if category != '전체':  # 전체 카테고리 제외
                try:
                    group_data = valid_data[valid_data['발주처_카테고리'] == category]['1순위사정률']
                    if len(group_data) >= 30:  # 30개 이상의 데이터가 있는 경우만 분석
                        mean = stats.loc[category, '평균 사정률']
                        std = stats.loc[category, '표준편차']
                        
                        intervals, prob_diffs = find_top_probabilities(group_data, mean, std)
                        
                        # 결과 저장
                        for i in range(min(5, len(intervals))):
                            stats.loc[category, f'{i+1}순위 구간'] = intervals[i]
                            stats.loc[category, f'{i+1}순위 초과확률(%)'] = prob_diffs[i]
                except Exception as e:
                    st.error(f"{category} 카테고리 분석 중 오류 발생: {str(e)}")
        
        # 전체 통계 계산 및 추가
        total_stats = pd.DataFrame({
            '건수': len(valid_data),
            '평균 사정률': valid_data['1순위사정률'].mean(),
            '최저 사정률': valid_data['1순위사정률'].min(),
            '최고 사정률': valid_data['1순위사정률'].max(),
            '중앙값 사정률': valid_data['1순위사정률'].median(),
            '표준편차': valid_data['1순위사정률'].std()
        }, index=['전체'])
        
        # 전체 데이터에 대한 정규분포 분석 수행
        try:
            mean = total_stats.loc['전체', '평균 사정률']
            std = total_stats.loc['전체', '표준편차']
            intervals, prob_diffs = find_top_probabilities(valid_data['1순위사정률'], mean, std)
            
            # 결과 저장
            for i in range(min(5, len(intervals))):
                total_stats.loc['전체', f'{i+1}순위 구간'] = intervals[i]
                total_stats.loc['전체', f'{i+1}순위 초과확률(%)'] = prob_diffs[i]
        except Exception as e:
            st.error(f"전체 통계 분석 중 오류 발생: {str(e)}")
            # 오류 발생 시 빈 값으로 설정
            for i in range(5):
                total_stats[f'{i+1}순위 구간'] = ''
                total_stats[f'{i+1}순위 초과확률(%)'] = ''
        
        # 전체 통계를 마지막 행으로 추가
        stats = pd.concat([stats, total_stats])
        
        # 통계 테이블 표시
        st.dataframe(
            stats.style.format({
                '건수': '{:,.0f}',
                '평균 사정률': '{:.4f}%',
                '최저 사정률': '{:.4f}%',
                '최고 사정률': '{:.4f}%',
                '중앙값 사정률': '{:.4f}%',
                '표준편차': '{:.4f}',
                '1순위 구간': '{}',
                '2순위 구간': '{}',
                '3순위 구간': '{}',
                '4순위 구간': '{}',
                '5순위 구간': '{}',
                '1순위 초과확률(%)': '{}',
                '2순위 초과확률(%)': '{}',
                '3순위 초과확률(%)': '{}',
                '4순위 초과확률(%)': '{}',
                '5순위 초과확률(%)': '{}'
            }).set_properties(subset=pd.IndexSlice['전체', :], 
                            **{'background-color': 'grey'}),
            use_container_width=True
        )
        
        return stats
        
    except Exception as e:
        st.error(f"카테고리별 통계 계산 중 오류 발생: {str(e)}")
        return None

# 한글 폰트 설정
def setup_korean_font():
    system_name = platform.system()
    
    if system_name == "Windows":
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system_name == "Darwin":  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
    
    plt.rcParams['axes.unicode_minus'] = False

# 특성 중요도 시각화
def plot_feature_importance(model, feature_importance_df):
    """특성 중요도를 시각화하는 함수"""
    try:
        # 한글 폰트 설정
        setup_korean_font()
        
        # 특성 중요도 정렬
        feature_importance = feature_importance_df.sort_values('중요도', ascending=True)
        
        # 상위 15개 특성만 선택
        top_features = feature_importance.tail(15)
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 수평 막대 그래프 그리기
        bars = ax.barh(range(len(top_features)), top_features['중요도'], 
                      color='steelblue', alpha=0.8)
        
        # y축 레이블 설정
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['특성'])
        
        # 막대 끝에 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', 
                   ha='left', va='center', fontsize=9)
        
        # 그래프 스타일링
        ax.set_title('특성 중요도', pad=20, fontsize=12)
        ax.set_xlabel('중요도', labelpad=10)
        
        # 그리드 추가
        ax.grid(True, axis='x', alpha=0.3)
        
        # 여백 조정
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"특성 중요도 시각화 중 오류 발생: {str(e)}")
        return None

# 예측 결과 시각화
def plot_prediction_distribution(predictions):
    # 한글 폰트 설정
    setup_korean_font()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 히스토그램 그리기
    n, bins, patches = ax.hist(predictions, bins=50, 
                             color='steelblue', alpha=0.8,
                             edgecolor='white')
    
    # 그래프 스타일링
    ax.set_title('예측된 1순위사정률 분포', pad=20)
    ax.set_xlabel('1순위사정률 (%)', labelpad=10)
    ax.set_ylabel('빈도수', labelpad=10)
    
    # x축 범위 설정
    ax.set_xlim(97, 103)
    
    # 그리드 추가
    ax.grid(True, alpha=0.3)
    
    # 여백 조정
    plt.tight_layout()
    
    return fig

########################################################

# 두 번째 셀 - 데이터 로드 및 전처리

# 문자열을 숫자로 변환하는 함수
def convert_to_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(',', '').strip()
        if '%' in x:
            return float(x.replace('%', '')) / 100
    return pd.to_numeric(x, errors='coerce')

def reclassify_small_categories(data, min_count=30):
    """카테고리 재분류하는 함수"""
    # 데이터 복사
    reclassified_data = data.copy()
    
    # 카테고리별 건수 계산
    category_counts = reclassified_data['발주처_카테고리'].value_counts()
    
    # 30건 미만인 카테고리 식별
    small_categories = category_counts[category_counts < min_count].index
    
    # 재분류 규칙 정의
    reclassification_rules = {
        # 기타공사로 통합
        '한국전력공사': '기타공사',
        '한국도로공사': '기타공사',
        '한국토지주택공사': '기타공사',  # 추가
        '한국마사회': '기타공사',
        
        # 기타공단으로 통합
        '국민건강보험공단': '기타공단',
        
        # 기타로 통합
        '협회조합': '기타',
        '공공기관': '기타',  # 추가
        
        # 기존 재분류 규칙 유지
        '대학교': '교육청',
        '학교': '교육청',
        '경찰소방': '공공기관',
        '연구기관': '공공기관',
        '의료기관': '공공기관'
    }
    
    # 먼저 30건 미만인 카테고리 처리
    for category in small_categories:
        if category in reclassification_rules:
            # 규칙이 있는 경우 해당 규칙 적용
            mask = reclassified_data['발주처_카테고리'] == category
            reclassified_data.loc[mask, '발주처_카테고리'] = reclassification_rules[category]
        else:
            # 규칙이 없는 경우 '기타'로 분류
            mask = reclassified_data['발주처_카테고리'] == category
            reclassified_data.loc[mask, '발주처_카테고리'] = '기타'
    
    # 나머지 재분류 규칙 적용
    for old_category, new_category in reclassification_rules.items():
        if old_category not in small_categories:  # 이미 처리된 카테고리는 제외
            mask = reclassified_data['발주처_카테고리'] == old_category
            reclassified_data.loc[mask, '발주처_카테고리'] = new_category
    
    return reclassified_data

def load_and_preprocess_data():
    try:
        # 데이터 폴더 경로 설정
        data_folder = os.path.dirname(os.path.abspath(__file__))
        
        # 입찰정보 파일 존재 여부 확인
        bid_files = [os.path.join(data_folder, f'bid_info_{i}.xlsx') for i in range(1, 6)]
        existing_bid_files = [f for f in bid_files if os.path.exists(f)]
        
        if not existing_bid_files:
            st.error("입찰정보를 찾 수 없습니다.")
            return pd.DataFrame()
        
        # 입찰정보 파일 읽기
        bid_dfs = []
        for file in existing_bid_files:
            try:
                df = pd.read_excel(file, header=1)
                if not df.empty:
                    bid_dfs.append(df)
            except Exception as e:
                continue
        
        if not bid_dfs:
            st.error("읽을 수 있는 입찰정보 파일이 없습니다.")
            return pd.DataFrame()
        
        # 입찰정보 병합 및 중복 제거
        merged_bid = pd.concat(bid_dfs, axis=0, ignore_index=True)
        
        # 중복 제거 전 데이터 수 저장
        before_drop_duplicates = len(merged_bid)
        
        # 공고번호 기준으로 중복 제거 (최신 데이터 유지)
        merged_bid = merged_bid.sort_values('개찰일', ascending=False)  # 최신 데이터가 위로 오도록 정렬
        merged_bid = merged_bid.drop_duplicates(subset=['공고번호'], keep='first')
        
        # 중복 제거 후 데이터 수 계산
        after_drop_duplicates = len(merged_bid)
        removed_duplicates = before_drop_duplicates - after_drop_duplicates
        
        if removed_duplicates > 0:
            st.info(f"중복된 공번호 {removed_duplicates}개가 제거되었습니다. (전체: {before_drop_duplicates}개 → {after_drop_duplicates}개)")
        
        try:
            # 낙찰정보 파일 존재 여부 확인 (영문 파일명 사용)
            award_files = [os.path.join(data_folder, f'award_info_{i}.xlsx') for i in range(1, 6)]
            existing_award_files = [f for f in award_files if os.path.exists(f)]
            
            award_dfs = []
            if existing_award_files:  # 낙찰정보 파일이 있는 경우에만 처리
                for file in existing_award_files:
                    try:
                        df = pd.read_excel(file, header=1)
                        if not df.empty:
                            award_dfs.append(df)
                    except Exception as e:
                        st.warning(f"낙찰정보 파일 '{file}' 읽기 실패: {str(e)}")
                        continue
            
            if award_dfs:  # 낙찰정보가 있는 경우에만 병합
                merged_award = pd.concat(award_dfs, axis=0, ignore_index=True)
                
                # 낙찰정보도 중복 제거 (최신 데이터 유지)
                before_award_duplicates = len(merged_award)
                merged_award = merged_award.drop_duplicates(subset=['공고번호'], keep='first')
                after_award_duplicates = len(merged_award)
                removed_award_duplicates = before_award_duplicates - after_award_duplicates
                
                if removed_award_duplicates > 0:
                    st.info(f"낙찰정보에서 중복된 공고번호 {removed_award_duplicates}개가 제거되었습니다.")
                
                # 입찰정보와 낙찰정보 병합
                columns_to_use = [col for col in merged_award.columns if col not in merged_bid.columns or col == '공고번호']
                merged_data = pd.merge(
                    merged_bid,
                    merged_award[columns_to_use],
                    on='공고번호',
                    how='left'
                )
            else:
                merged_data = merged_bid
                st.info("낙찰정보 파일이 없거나 읽을 수 없어 입찰정보만 처리합니다.")
        
        except Exception as e:
            st.warning(f"낙찰정보 처리 중 오류 발생: {str(e)}")
            merged_data = merged_bid
        
        # 발주처 카테고리 추가 및 재분류를 한 번만 수행
        merged_data['발주처_카테고리'] = merged_data['발주처'].apply(categorize_client)
        merged_data = reclassify_small_categories(merged_data)
        
        # 숫자형 변환
        numeric_columns = ['기초금액', '추정가격', '투찰률', 'A값', '순공사원가', '1순위사정률']
        for col in numeric_columns:
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].apply(convert_to_numeric)
        
        return merged_data
        
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        return pd.DataFrame()

# 데이터 로드 및 전처리 실행
processed_data = load_and_preprocess_data()

########################################################
# 웹에 데이터 프레임 표시
st.title('나라장터 투찰을 위한 사정률 구하기')

# 데이터 프레임 표시    
st.header("1. 2021년 01월 이후 나라장터 입찰 및 낙찰 정보", divider=True)

st.dataframe(processed_data, use_container_width=False)
# 데이터 정보 표시
st.caption(f"전체 {len(processed_data):,}개의 데이터가 포함되어 있습니다.")

########################################################
# 파일 업로드 섹션들을 나누어 배치
col1, col2 = st.columns(2)

with col1:
    st.subheader("새로운 입찰 데이터 업로드")
    bid_file = st.file_uploader("입찰 데이터 파일을 업로드하세요", type=['xlsx', 'xls'], key='bid_upload')
    
    if bid_file is not None:
        try:
            # 기존의 입찰 데이터 업로드 로직
            new_data = pd.read_excel(bid_file, header=1)
            existing_columns = processed_data.columns
            new_data = new_data[existing_columns.intersection(new_data.columns)]
            
            # # 데이터 확인을 위한 정보 출력
            # st.write("업로드된 파일의 열:", new_data.columns.tolist())
            
            required_columns = ['공고번호'] + ['기초금액', '추정가격', '투찰률', 'A값', '순공사원가']
            missing_columns = set(required_columns) - set(new_data.columns)
            
            if missing_columns:
                st.error(f"필수 열이 누락되었습니다: {', '.join(missing_columns)}")
            else:
                # 업데이트 전 데이터 상태 확인
                before_count = len(processed_data)
                before_empty = processed_data['1순위사정률'].isna().sum()
                
                # 숫자형 데이터 변환
                numeric_columns = ['기초금액', '추정가격', '투찰률', 'A값', '순공사원가']
                for col in numeric_columns:
                    if col in new_data.columns:
                        new_data[col] = new_data[col].apply(convert_to_numeric)
                
                # 기존 데이터와 병합 (중복 제거)
                processed_data = pd.concat([processed_data, new_data], axis=0)
                processed_data = processed_data.drop_duplicates(subset=['공고번호'], keep='first')
                
                # 업데이트 후 데이터 상태 확인
                after_count = len(processed_data)
                after_empty = processed_data['1순위사정률'].isna().sum()
                update_count = after_count - before_count
                
                st.success(f"""
                입찰 데이터가 성공적으로 업데이트되었습니다.
                - 추가된 새로운 행 수: {update_count}개
                - 업데이트 전 전체 행 수: {before_count}개
                - 업데이트 후 전체 행 수: {after_count}개
                - 업데이트 전 빈 값: {before_empty}개
                - 업데이트 후 빈 값: {after_empty}개
                """)
                
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
            st.write("오류 상세:", str(e))

with col2:
    st.subheader("새로운 낙찰 데이터 업로드")
    award_file = st.file_uploader("낙찰 데이터 파일을 업로드하세요", type=['xlsx', 'xls'], key='award_upload')
    
    if award_file is not None:
        try:
            # 새로운 낙찰 데이터 읽기 (header=1 설정)
            new_award_data = pd.read_excel(award_file, header=1)
            
            # # 데이터 확인을 위한 정보 출력
            # st.write("업로드된 파일의 열:", new_award_data.columns.tolist())
            
            # 필수 열 확인
            required_columns = ['공고번호', '1순위사정률']
            missing_columns = set(required_columns) - set(new_award_data.columns)
            
            if missing_columns:
                st.error(f"필수 열이 누락되었습니다: {', '.join(missing_columns)}")
            else:
                # 숫자형 이터 변환
                if '1순위사정률' in new_award_data.columns:
                    new_award_data['1순위사정률'] = new_award_data['1순위사정률'].apply(convert_to_numeric)
                
                # 업데이트 전 데이터 상태 확인
                before_empty = processed_data['1순위사정률'].isna().sum()
                
                # 기존 데이터와 새로운 데이터의 공고번호 매칭
                update_count = 0
                for idx, row in new_award_data.iterrows():
                    mask = (processed_data['공고번호'] == row['공고번호']) & \
                          (pd.isna(processed_data['1순위사정률']))
                    
                    if mask.any():
                        # 빈 값만 업데이트
                        for col in new_award_data.columns:
                            if col in processed_data.columns and col != '공고번호':
                                processed_data.loc[mask, col] = row[col]
                        update_count += 1
                
                # 업데이트 후 데이터 상태 확인
                after_empty = processed_data['1순위사정률'].isna().sum()
                
                st.success(f"""
                낙찰 데이터가 성공적으로 업데이트되었습니다.
                - 업데이트된 행 수: {update_count}개
                - 업데이트 전 빈 값: {before_empty}개
                - 업데이트 후 빈 값: {after_empty}개
                """)
                
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
            st.write("오류 상세:", str(e))

########################################################
# 엑셀 다운로드 기능
st.text("\n=== 데이터 다운로드 ===")
# BytesIO를 사용하여 메모리에 엑셀 파일 생성
from io import BytesIO
import xlsxwriter

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        # 자동 열 너비 조정
        worksheet = writer.sheets['Sheet1']
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)
    output.seek(0)
    return output

# 다운로드 버튼 생성
excel_file = to_excel(processed_data)
st.download_button(
    label="📥 업데이트된 데이터 엑셀 다운로드",
    data=excel_file,
    file_name='나라장터_입찰정보.xlsx',
    mime='application/vnd.ms-excel'
)

# 데이터 정보 표시
st.caption(f"전체 {len(processed_data):,}개의 데이터가 포함되어 있습니다.")

########################################################
# 발주처 카테고리별 통계 표시
st.header("2. 발주처 카테고리별 통계", divider=True)

# 전체 데이터의 카테고리별 건수
total_counts = processed_data['발주처_카테고리'].value_counts()
# st.text("\n=== 전체 발주처 카테고리별 건수 ===")
# for category, count in total_counts.items():
#     st.text(f"{category}: {count:,}건")

# 1순위사정률이 있는 데이터의 통계
st.subheader("발주처 카테고리별 1순위사정률 통계")
category_stats = show_category_statistics(processed_data)

########################################################
st.header("3. 사정률 예측 모델", divider=True)

# 모델 학습에 사용할 완전한 데이터 확인
complete_data = processed_data.dropna(subset=['기초금액', '추정가격', '투찰률', 'A값', '순공사원가', '1순위사정률'])

# 머신러닝 학습 버튼
if st.button("🤖 머신러닝 모델 학습 시작", type="primary"):
    with st.spinner("머신러닝 모델 학습 중..."):
        try:
            # 예측 모델 초기화 및 학습
            predictor = BidPricePredictor(processed_data)
            model, valid_data = predictor.train_model()
            
            if model is not None and hasattr(predictor, 'X_train') and hasattr(predictor, 'y_train'):
                # 성능 평가
                train_score = model.score(predictor.X_train, predictor.y_train)
                test_score = model.score(predictor.X_test, predictor.y_test)
                
                # 성능 지표 표시
                st.text("\n=== 모델 성능 평가 ===")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="학습 데이터 R² 점수", value="{:.4f}".format(train_score))
                with col2:
                    st.metric(label="테스트 데이터 R² 수치", value="{:.4f}".format(test_score))
                
                # 학습된 모델을 세션 상태에 저장
                st.session_state['predictor'] = predictor
                st.session_state['model_trained'] = True
                
                st.success("모델 학습이 완료되었습니다! 이제 새로운 입찰건의 사정률을 예측할 수 있습니다.")
            else:
                st.error("모델 학습에 실패했습니다.")
                st.session_state['model_trained'] = False
            
        except Exception as e:
            st.error(f"모델 학습 중 오류가 발생했습니다: {str(e)}")
            st.session_state['model_trained'] = False

# 예측 실행 (모델이 학습된 경우에만)
if st.session_state.get('model_trained', False):
    # 예측할 데이터 선택
    prediction_candidates = processed_data[pd.isna(processed_data['1순위사정률'])].copy()
    prediction_data = prediction_candidates.dropna(subset=['기초금액', '추정가격', '투찰률', 'A값', '순공사원가', '발주처'])
    
    if len(prediction_data) > 0:
        st.subheader("새로운 입찰건 사정률 예측 결과")
        
        # 발주처 카테고리는 원본 데이터에서 가져오기
        prediction_data['발주처_카테고리'] = prediction_data['발주처'].apply(categorize_client)
        
        # 카테고리별 통계 정보 가져오기
        # category_stats = show_category_statistics(processed_data)
        
        # 예측 데이터에 카테고리별 순위 정보 추가
        for i in range(1, 6):
            prediction_data[f'{i}순위 구간'] = prediction_data['발주처_카테고리'].map(
                category_stats[f'{i}순위 구간']
            )
            prediction_data[f'{i}순위 초과확률(%)'] = prediction_data['발주처_카테고리'].map(
                category_stats[f'{i}순위 초과확률(%)']
            )
        
        # 예측 실행
        predictor = st.session_state['predictor']
        predictions = []
        
        for idx, row in prediction_data.iterrows():
            try:
                predicted_rate = predictor.predict_rate(row.to_dict())
                predictions.append(predicted_rate)
            except Exception as e:
                st.error(f"예측 오류 (공고번호: {row.get('공고번호', 'N/A')}): {str(e)}")
                predictions.append(np.nan)
        
        # 예측 결과를 데이터프레임에 추가
        prediction_data['사정률(ML예측값)'] = predictions
        
        # 표시할 열 선택 및 정렬
        display_columns = [
            '공고번호',
            '공사명',
            '발주처',
            '발주처_카테고리',
            '기초금액',
            '추정가격',
            '투찰률',
            'A값',
            '순공사원가',
            '사정률(ML예측값)',
            '1순위 구간',
            '1순위 초과확률(%)',
            '2순위 구간',
            '2순위 초과확률(%)',
            '3순위 구간',
            '3순위 초과확률(%)',
            '4순위 구간',
            '4순위 초과확률(%)',
            '5순위 구간',
            '5순위 초과확률(%)'
        ]
        
        # 실제 표시할 열은 데이터프레임에 존재하는 열만 선택
        display_columns = [col for col in display_columns if col in prediction_data.columns]
        
        # 데이터프레임 표시
        st.dataframe(
            prediction_data[display_columns].style.format({
                '공고번호': '{}',
                '공사명': '{}',
                '발주처': '{}',
                '발주처_카테고리': '{}',
                '기초금액': '{:,.0f}',
                '추정가격': '{:,.0f}',
                '투찰률': '{:.4f}',
                'A값': '{:.4f}',
                '순공사원가': '{:,.0f}',
                '사정률(ML예측값)': '{:.4f}',
                '1순위 구간': '{}',
                '2순위 구간': '{}',
                '3순위 구간': '{}',
                '4순위 구간': '{}',
                '5순위 구간': '{}',
                '1순위 초과확률(%)': '{}',
                '2순위 초과확률(%)': '{}',
                '3순위 초과확률(%)': '{}',
                '4순위 초과확률(%)': '{}',
                '5순위 초과확률(%)': '{}'
            }),
            use_container_width=True
        )

else:
    st.info("먼저 머신러닝 모델 학습을 실행해주세요.")
