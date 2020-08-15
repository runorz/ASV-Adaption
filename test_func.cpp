#include<iostream>
#include<cmath>

double compute_angle(double asv_x, double asv_y, double waypoint_x, double waypoint_y){
  double d_x = waypoint_x - asv_x;
  double d_y = waypoint_y - asv_y;
  if(d_x <= 0){
    return acos(d_y/(sqrt(d_x*d_x+d_y*d_y) * sqrt(1)));
  }else{
    return -acos(d_y/(sqrt(d_x*d_x+d_y*d_y) * sqrt(1)));
  }
}

int main(){

    std::cout << compute_angle(0, 1, 1, 0) << std::endl;
    return 0;

}