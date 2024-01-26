import 'dart:convert';
import 'dart:io';

import 'package:flutter/cupertino.dart';
import 'package:http/http.dart' as http;
import 'package:dio/dio.dart';
const URL = "http://10.0.2.2:5000";
const Upload_Image_URL = "$URL/uploadImage";

uploadImage(File Image, String type_process) async {
  final dio = Dio();
  FormData formData = FormData.fromMap({
      'image': await MultipartFile.fromFile(
        Image.path,
        filename: Image.path.split('/').last,
      ),
      'type': type_process,
    });
    
    Response response = await dio.post(
      Upload_Image_URL,
      data: formData,
      options: Options(
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      ),
    );
    try{
 if (response.statusCode == 200) {
      if (response.data['status'] == "200") {
        return response.data['data'].toString();
      } else {
        throw Exception('Failed to upload image');
      }
    } else {
      throw Exception('Failed to upload image');
    }
  } catch (e) {
    print('Error: $e');
    return null;
  }
  // final request = http.MultipartRequest('POST', Uri.parse(Upload_Image_URL));
  // final headers = {'Content-Type': 'multipart/form-data'};
  // request.files.add(http.MultipartFile(
  //     'image', Image.readAsBytes().asStream(), Image.lengthSync(),
  //     filename: Image.path.split('/').last));
  // request.fields['type'] = type_process;

  // request.headers.addAll(headers);
  // final response = await request.send();
  // try {
  //   http.Response res = await http.Response.fromStream(response);
  //   if (jsonDecode(res.body)['status'] != "200") {
  //     throw Exception('Failed to upload image');
  //   }
  //   else{
  //     return (jsonDecode(res.body)['data'].toString());
  //   }
  // } catch (e) {
  //   print('Error: $e');
  // }
}
Widget buildTable(String text) {
    List<List<String>> rows = text
        .split('\n')
        .where((row) => row.trim().isNotEmpty)
        .map((row) => row.split('|').where((cell) => cell.trim().isNotEmpty).toList())
        .toList();

    if (rows.isNotEmpty) {
      int numColumns = rows[0].length;
      Map<int, TableColumnWidth> columnWidths = {};
      for (int i = 0; i < numColumns; i++) {
        columnWidths[i] = const FixedColumnWidth(80.0); // Set your default width here
      }
      return Table(
        columnWidths: columnWidths,
        border: TableBorder.all(),
        children: rows.map((row) {
          return TableRow(
            children: row.map((cell) {
              return TableCell(
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Center(child: Text(cell.trim(), style: const TextStyle(fontSize: 20.0,fontWeight: FontWeight.bold))),
                ),
              );
            }).toList(),
          );
        }).toList(),
      );
    } else {
      return Container(); 
    }
  }