from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, stddev, lit, abs as abs_spark, sum as sum_spark, when, isnan, collect_list
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Khởi tạo Spark Session
spark = SparkSession.builder \
    .appName("CollaborativeFiltering") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

class CollaborativeFiltering:
    """
    Lớp triển khai thuật toán Collaborative Filtering sử dụng hệ số tương quan Pearson
    để đo độ tương đồng giữa các người dùng và gợi ý sản phẩm với PySpark.
    """

    def __init__(self, N, spark_df):
        """
        Hàm khởi tạo cho CollaborativeFiltering

        Tham số:
        - N: Số lượng người dùng tương tự cần xem xét
        - spark_df: Spark DataFrame chứa dữ liệu ratings
        """
        self.N = N
        self.spark = spark_df.sparkSession
        self.data = spark_df
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.valid_users = None
        self.user_means = None

        # Loại bỏ users không khách quan
        self._remove_constant_rating_users()

        # Xây dựng ma trận user-item
        self._build_user_item_matrix()

        # Calculate user means for normalization
        self._calculate_user_means()

        # Tính ma trận độ tương đồng
        self._calculate_user_similarity()

    def _remove_constant_rating_users(self):
        """
        Loại bỏ các users có constant ratings (đánh giá tất cả items với cùng một rating value)
        """
        # Tính độ lệch chuẩn của ratings cho mỗi user
        user_ratings_std = self.data.groupBy("user").agg(stddev("rating").alias("std_rating"))
        
        # Tìm users có std = 0 (constant ratings) hoặc NULL (chỉ có 1 rating)
        constant_users = user_ratings_std.filter(
            (col("std_rating") == 0) | col("std_rating").isNull()
        ).select("user").rdd.flatMap(lambda x: x).collect()

        # Lấy tất cả user IDs từ DataFrame gốc
        all_users = self.data.select("user").distinct().rdd.flatMap(lambda x: x).collect()
        
        # Lưu lại danh sách users hợp lệ
        self.valid_users = [u for u in all_users if u not in constant_users]

        # Cập nhật data chỉ giữ lại valid users
        self.data = self.data.filter(col("user").isin(self.valid_users))

    def _build_user_item_matrix(self):
        """
        Xây dựng ma trận user-item từ dữ liệu
        """
        # PySpark không có hàm pivot trực tiếp như pandas
        # Chúng ta sẽ giữ dữ liệu trong dạng user-item-rating
        # và sử dụng các phép join khi cần
        self.user_item_matrix = self.data

    def _calculate_user_means(self):
        """
        Calculate mean rating for each user for normalization
        """
        self.user_means = self.data.groupBy("user").agg(avg("rating").alias("mean_rating"))

    def _calculate_user_similarity(self):
        """
        Tính ma trận độ tương đồng giữa người dùng sử dụng hệ số tương quan Pearson
        với dữ liệu đã được chuẩn hóa
        """
        # Lấy danh sách users
        users = self.valid_users
        n_users = len(users)

        # Tạo cấu trúc để lưu ma trận similarity
        similarity_rows = []

        # Join dữ liệu với user means để chuẩn hóa ratings
        normalized_ratings = self.data.join(
            self.user_means,
            on="user",
            how="inner"
        ).withColumn(
            "normalized_rating", 
            col("rating") - col("mean_rating")
        )

        # Chuyển normalized_ratings thành RDD để tính toán hiệu quả hơn
        # Mỗi phần tử có dạng ((user, item), normalized_rating)
        normalized_ratings_rdd = normalized_ratings.select(
            "user", "item", "normalized_rating"
        ).rdd.map(
            lambda row: ((row.user, row.item), row.normalized_rating)
        ).cache()

        # Tính pearson correlation
        # Đây là phần tính toán chính, chúng ta sẽ sử dụng map-reduce pattern
        # Vì cần tính tương quan giữa từng cặp user, và PySpark không có hàm pearsonr sẵn,
        # chúng ta cần một cách tiếp cận khác

        # Đầu tiên, thu thập ratings của mỗi user
        user_ratings_dict = {}
        for user in users:
            user_data = normalized_ratings.filter(col("user") == user).select(
                "item", "normalized_rating"
            ).rdd.collectAsMap()
            user_ratings_dict[user] = user_data

        # Sau đó, tính toán tương quan cho mỗi cặp user
        for i, user_i in enumerate(users):
            row = {user_j: 0.0 for user_j in users}
            row[user_i] = 1.0  # User hoàn toàn tương đồng với chính họ
            
            for j, user_j in enumerate(users):
                if i < j:  # Chỉ tính nửa trên của ma trận (ma trận đối xứng)
                    ratings_i = user_ratings_dict[user_i]
                    ratings_j = user_ratings_dict[user_j]
                    
                    # Tìm các sản phẩm được đánh giá bởi cả hai người dùng
                    common_items = set(ratings_i.keys()) & set(ratings_j.keys())
                    
                    if len(common_items) > 1:  # Cần ít nhất 2 sản phẩm chung
                        # Get ratings
                        user_i_ratings = [ratings_i[item] for item in common_items]
                        user_j_ratings = [ratings_j[item] for item in common_items]
                        
                        # Calculate standard deviations
                        std_i = float(np.std(user_i_ratings))
                        std_j = float(np.std(user_j_ratings))
                        
                        # Check if either array is constant after normalization
                        if std_i > 1e-10 and std_j > 1e-10:  # Using a small epsilon instead of 0
                            # Tính hệ số tương quan Pearson
                            try:
                                correlation, _ = pearsonr(user_i_ratings, user_j_ratings)
                                if not np.isnan(correlation):
                                    # Chuyển đổi numpy.float64 thành float Python thông thường
                                    row[user_j] = float(correlation)
                                    # Vì ma trận đối xứng
                            except Exception:
                                pass  # Xử lý lỗi, giữ nguyên giá trị 0
            
            similarity_rows.append(row)

        # Chuyển đổi về dạng DataFrame
        similarity_schema = StructType([
            StructField("user_i", IntegerType(), False)
        ] + [
            StructField(f"user_{user}", FloatType(), True) for user in users
        ])
        
        # Tạo DataFrame similarity
        similarity_data = [(user_i,) + tuple(row[user_j] for user_j in users) 
                          for user_i, row in zip(users, similarity_rows)]
        
        # Tạo DataFrame từ dữ liệu
        # Sử dụng createDataFrame thay vì createDataset vì không cần type inference
        self.user_similarity_matrix = spark.createDataFrame(similarity_data, 
                                                          schema=similarity_schema)

    def predict(self, user_id, num_recommendations):
        """
        Dự đoán và gợi ý các sản phẩm cho người dùng
        sử dụng dữ liệu chuẩn hóa

        Tham số:
        - user_id: ID của người dùng
        - num_recommendations: Số lượng sản phẩm cần gợi ý

        Trả về:
        - DataFrame với các sản phẩm được sắp xếp theo điểm dự đoán giảm dần
        """
        # Kiểm tra xem user có trong danh sách valid users không
        if user_id not in self.valid_users:
            print(f"User {user_id} has constant ratings or not in training data")
            return self.spark.createDataFrame([], schema=StructType([
                StructField("user", IntegerType(), True),
                StructField("item", IntegerType(), True),
                StructField("predicted_rating", FloatType(), True)
            ]))

        # Lấy mean của user
        user_mean = self.user_means.filter(col("user") == user_id).select("mean_rating").collect()[0]["mean_rating"]
        
        # Lấy tất cả items mà user đã đánh giá
        user_rated_items = self.data.filter(col("user") == user_id).select("item").rdd.flatMap(lambda x: x).collect()
        
        # Lấy các items chưa được đánh giá (hiệu của tất cả items với items đã đánh giá)
        all_items = self.data.select("item").distinct().rdd.flatMap(lambda x: x).collect()
        unrated_items = [item for item in all_items if item not in user_rated_items]
        
        # Lấy N người dùng tương tự nhất
        # Lấy hàng tương ứng với user hiện tại trong ma trận similarity
        similarity_row = self.user_similarity_matrix.filter(col("user_i") == user_id).first()
        
        if similarity_row is None:
            return self.spark.createDataFrame([], schema=StructType([
                StructField("user", IntegerType(), True),
                StructField("item", IntegerType(), True),
                StructField("predicted_rating", FloatType(), True)
            ]))
        
        # Chuyển đổi similarity_row thành Dictionary để dễ sắp xếp
        similarity_dict = {user: similarity_row[f"user_{user}"] for user in self.valid_users if user != user_id}
        
        # Sắp xếp theo similarity và lấy N người dùng tương tự nhất
        top_similar_users = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)[:self.N]
        top_similar_users = [user for user, _ in top_similar_users]
        
        # Dự đoán rating cho mỗi item chưa được đánh giá
        predictions = []
        
        # Lấy ratings của top_similar_users cho các sản phẩm chưa đánh giá
        similar_users_ratings = self.data.filter(
            col("user").isin(top_similar_users) & col("item").isin(unrated_items)
        ).join(
            self.user_means, 
            on="user",
            how="inner"
        ).withColumn(
            "normalized_rating", 
            col("rating") - col("mean_rating")
        )
        
        # Lấy similarity của mỗi user trong top_similar_users
        similar_users_similarities = {user: similarity_dict[user] for user in top_similar_users}
        
        # Tính weighted sum cho mỗi item
        for item in unrated_items:
            item_ratings = similar_users_ratings.filter(col("item") == item).collect()
            
            if len(item_ratings) > 0:
                weighted_sum = 0.0
                similarity_sum = 0.0
                
                for row in item_ratings:
                    similarity = similar_users_similarities[row.user]
                    weighted_sum += similarity * row.normalized_rating
                    similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating_norm = weighted_sum / similarity_sum
                    predicted_rating = predicted_rating_norm + user_mean
                    
                    predictions.append((user_id, item, float(predicted_rating)))
        
        # Tạo DataFrame chứa predictions
        if predictions:
            predictions_df = self.spark.createDataFrame(
                predictions,
                ["user", "item", "predicted_rating"]
            )
            
            # Sắp xếp theo predicted_rating giảm dần và lấy top num_recommendations
            predictions_df = predictions_df.orderBy(col("predicted_rating").desc()).limit(num_recommendations)
            return predictions_df
        else:
            return self.spark.createDataFrame([], schema=StructType([
                StructField("user", IntegerType(), True),
                StructField("item", IntegerType(), True),
                StructField("predicted_rating", FloatType(), True)
            ]))

def calculate_rmse_manual(actual_ratings, predicted_ratings):
    """
    Tính RMSE
    
    Tham số:
    - actual_ratings: List hoặc numpy array chứa giá trị thực tế
    - predicted_ratings: List hoặc numpy array chứa giá trị dự đoán
    
    Trả về:
    - Giá trị RMSE
    """
    # Chuyển đổi sang numpy array nếu cần thiết
    actual_arr = np.array(actual_ratings, dtype=float)
    predicted_arr = np.array(predicted_ratings, dtype=float)
    
    # Tính toán RMSE
    errors = actual_arr - predicted_arr
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    
    # Chuyển đổi từ numpy.float64 sang float thông thường
    return float(rmse)

def analyze_constant_rating_users(spark_df):
    """
    Phân tích và thống kê users có constant ratings
    
    Tham số:
    - spark_df: Spark DataFrame chứa dữ liệu ratings
    """
    # Tính độ lệch chuẩn của ratings cho mỗi user
    user_ratings_std = spark_df.groupBy("user").agg(
        stddev("rating").alias("std_rating"),
        count("rating").alias("rating_count")
    )
    
    # Tìm users có std = 0 (constant ratings) hoặc NULL (chỉ có 1 rating)
    constant_users_df = user_ratings_std.filter(
        (col("std_rating") == 0) | col("std_rating").isNull()
    )
    
    # Thống kê
    total_users = spark_df.select("user").distinct().count()
    constant_users_count = constant_users_df.count()
    percentage = (constant_users_count / total_users) * 100
    
    print(f"Tổng số lượng người dùng: {total_users}")
    print(f"Số người dùng đánh giá không khách quan: {constant_users_count}")
    print(f"Tỉ lệ phần trăm: {percentage:.2f}%")
    
    # Hiển thị ví dụ
    print("\n Các người dùng đánh giá không khách quan:")
    
    # Join để lấy giá trị rating
    constant_users_with_ratings = constant_users_df.join(
        spark_df,
        on="user",
        how="inner"
    ).select("user", "rating_count", "rating").distinct()
    
    # Lấy 80 người dùng đầu tiên
    constant_users_sample = constant_users_with_ratings.limit(80).collect()
    
    for row in constant_users_sample:
        print(f"Người dùng {row.user}: {row.rating_count} lượt đánh giá, giá trị {row.rating}")
    
    # Trả về danh sách user IDs
    return constant_users_df.select("user").rdd.flatMap(lambda x: x).collect()

def main():
    # Định nghĩa schema cho DataFrame
    ratings_schema = StructType([
        StructField("index", IntegerType(), True),
        StructField("user", IntegerType(), True),
        StructField("item", IntegerType(), True),
        StructField("rating", FloatType(), True)
    ])
    
    # Load dataset
    data = spark.read.csv("ratings2k.csv", header=True, schema=ratings_schema)
    
    print(f"Cấu trúc của dataset: {data.count()} hàng x {len(data.columns)} cột")
    print(f"Các trường của dataset: {data.columns}")
    
    # Phân tích users có constant ratings
    constant_users = analyze_constant_rating_users(data)
    
    # Chia dataset thành training và test sets với tỷ lệ 8:2
    # PySpark không có hàm train_test_split như scikit-learn
    # Chúng ta sẽ sử dụng randomSplit
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Training set size: {train_data.count()}")
    print(f"Test set size: {test_data.count()}")
    
    # Đánh giá thuật toán với N trong khoảng [2, 16]
    print("\nĐánh giá thuật toán với N từ 2 đến 16...")
    N_values = list(range(2, 17))
    rmse_values = []
    
    for N in N_values:
        print(f"\nN = {N}")
        
        # Khởi tạo model với training data
        cf_model = CollaborativeFiltering(N, train_data)
        
        # Dự đoán cho test set
        test_users = test_data.select("user").distinct().rdd.flatMap(lambda x: x).collect()
        all_predictions = []
        
        for user in test_users:
            # Bỏ qua users có constant ratings
            if user not in cf_model.valid_users:
                continue
            
            # Lấy tất cả ratings của user trong test set
            user_test_ratings = test_data.filter(col("user") == user)
            user_test_count = user_test_ratings.count()
            
            # Dự đoán cho các items trong test set
            predictions = cf_model.predict(user, user_test_count)
            
            if predictions.count() > 0:
                # Chỉ giữ lại các predictions cho items có trong test set
                merged = predictions.join(user_test_ratings, on=["user", "item"], how="inner")
                
                if merged.count() > 0:
                    all_predictions.append(merged)
        
        # Tính RMSE
        if all_predictions:
            # Union tất cả DataFrames trong all_predictions
            all_predictions_df = all_predictions[0]
            for df in all_predictions[1:]:
                all_predictions_df = all_predictions_df.union(df)
            
            if all_predictions_df.count() > 0:
                # Collect data về driver để tính RMSE
                actuals = all_predictions_df.select("rating").rdd.flatMap(lambda x: x).collect()
                predicteds = all_predictions_df.select("predicted_rating").rdd.flatMap(lambda x: x).collect()
                
                rmse = calculate_rmse_manual(actuals, predicteds)
                rmse_values.append(rmse)
                print(f"RMSE: {rmse:.4f}")
            else:
                rmse_values.append(None)
                print("Không có giá trị dự đoán hợp lệ")
        else:
            rmse_values.append(None)
            print("Không dự đoán được")
    
    # Vẽ bar chart cho RMSE values với mỗi N
    valid_indices = [i for i, rmse in enumerate(rmse_values) if rmse is not None]
    valid_N = [N_values[i] for i in valid_indices]
    valid_rmse = [float(rmse_values[i]) for i in valid_indices]  # Chuyển đổi sang float thông thường
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid_N, valid_rmse, color='skyblue', edgecolor='black')
    plt.xlabel('N (Số lượng người dùng cần xét)')
    plt.ylabel('RMSE')
    plt.title('Biểu đồ so sánh RMSE với các giá trị N khác nhau')
    plt.xticks(valid_N)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, rmse in zip(bars, valid_rmse):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{rmse:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    best_N_index = int(np.argmin(valid_rmse))
    best_N = valid_N[best_N_index]
    best_rmse = float(valid_rmse[best_N_index])
    print(f"\nVới N = {best_N} có giá trị RMSE tốt nhất = {best_rmse:.4f}")
    
    # Tạo model cuối cùng với toàn bộ dataset để deploy
    final_model = CollaborativeFiltering(best_N, data)
    
    # Demo the prediction functionality
    print("\n--- DEMO PREDICTION FOR SPECIFIC USERS ---")
    
    # Choose a few users to demonstrate (pick from valid users)
    demo_users = final_model.valid_users[:3]  # Get first 3 valid users
    
    for user_id in demo_users:
        print(f"\nRecommendations for User {user_id}:")
        
        # Get items user has already rated
        user_rated_items = data.filter(col("user") == user_id)
        user_rated_count = user_rated_items.count()
        print(f"User has already rated {user_rated_count} items")
        
        if user_rated_count > 0:
            print("Sample of user's ratings:")
            user_rated_items.select("user", "item", "rating").show(3)
        
        # Get recommendations for this user
        recommendations = final_model.predict(user_id, 5)
        
        if recommendations.count() > 0:
            print("\nTop 5 recommended items:")
            recommendations.select("item", "predicted_rating").show()
        else:
            print("No recommendations available for this user.")
    
    return final_model

if __name__ == "__main__":
    model = main()
    # Dừng Spark Session khi hoàn thành
    spark.stop()