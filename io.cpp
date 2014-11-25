#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(1073741824, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

int get_group_num(int label) {
   if (label <=20) {
      return 0;
   } else if(label <=29) {
      return 1;
   } else if(label <=39) {
      return 2;
   } else if(label <=59) {
      return 3;
   } else {
      return 4;
   }
}

int label_grouping(int img_label, int base_label) {
   /*
   if (get_group_num(img_label) == get_group_num(base_label)) {
      return 0;
   } else if (get_group_num(img_label) > get_group_num(base_label)) {
      return 1;
   } else {
      return 2;
   } */
   if (get_group_num(img_label) > get_group_num(base_label)) {
      return 1; 
   } else {
      return 0;
   }
}

bool ReadImagePairToDatum(const string& file1, const int label1, 
  const string& file2, const int label2, const int height, const int width,
  const bool is_color, Datum* datum) {
  cv::Mat cv_img  = ReadImageToCVMat(file1, height, width, is_color);
  cv::Mat cv_base = ReadImageToCVMat(file2, height, width, is_color);
  if (cv_img.data && cv_base.data) {
    PairCVMatToDatum(cv_img, cv_base, datum);
    datum->set_label(label_grouping(label1, label2));
    return true;
  } else {
    return false;
  }
}

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

cv::Mat DecodeDatumToCVMat(const Datum& datum,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imdecode(cv::Mat(vec_data), cv_read_flag);
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv::imdecode(vec_data, cv_read_flag);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// if height and width are set it will resize it
// If Datum is not encoded will do nothing
bool DecodeDatum(const int height, const int width, const bool is_color,
                Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), height, width, is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

void PairCVMatToDatum(const cv::Mat& cv_img1, const cv::Mat& cv_img2, Datum* datum) {
   //LOG(ERROR) << "PairCVMatToDatum is called";
  CHECK(cv_img1.depth() == CV_8U) << "Image 1 data type must be unsigned byte";
  CHECK(cv_img2.depth() == CV_8U) << "Image 2 data type must be unsigned byte";
  CHECK(cv_img1.size() == cv_img2.size()) << "Image 1 & 2 are not equal size";
  const int img_num = 2;
  // tricks here
  // one channel for each image in the pair (they should be same size)
  // datum->set_channels(cv_img.channels());
  int img_channels = cv_img1.channels();
  datum->set_channels(img_num * img_channels);
  datum->set_height(cv_img1.rows);
  datum->set_width(cv_img1.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int single_img_size = cv_img1.channels()*cv_img1.cols*cv_img1.rows;

  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  
  CHECK(datum_size == 2 * single_img_size) << "Datum is not equal 2 times of images!";

//  LOG(ERROR) << "img ch: " << img_channels 
//    << "\tdataum ch: " << datum_channels;

  std::string buffer(datum_size, ' ');
  int datum_index;
  // first image
     for (int h = 0; h < datum_height; ++h) {
        const uchar* ptr = cv_img1.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < datum_width; ++w) {
          for (int c = 0; c < img_channels; ++c) {
            int datum_index = (c * datum_height + h) * datum_width + w;
            buffer[datum_index] = static_cast<char>(ptr[img_index++]);
          }
        }
     }
   const int one_image_offset = img_channels * datum_height * datum_width;
  // LOG(ERROR) <<"Offset: " << one_image_offset << " = "
  //    << datum_height << " * " << datum_width << " * " << img_channels;
  // second image
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img2.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
         int datum_index = (c * datum_height + h) * datum_width + w 
                            + one_image_offset;
         buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
 }
  datum->set_data(buffer);
}

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the dataset exists.
  CHECK(H5LTfind_dataset(file_id, dataset_name_))
      << "Failed to find HDF5 dataset " << dataset_name_;
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

  blob->Reshape(
    dims[0],
    (dims.size() > 1) ? dims[1] : 1,
    (dims.size() > 2) ? dims[2] : 1,
    (dims.size() > 3) ? dims[3] : 1);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string dataset_name, const Blob<float>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
    const hid_t file_id, const string dataset_name, const Blob<double>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

}  // namespace caffe
