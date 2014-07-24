namespace eigenpy
{
  typedef Eigen::Quaternion<double,Eigen::DontAlign> Quaterniond_fx;
  //typedef Eigen::AngleAxis<double> AngleAxis_fx;

  void exposeQuaternion();
  void exposeAngleAxis();
} // namespace eigenpy
