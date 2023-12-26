import 'package:bloc/bloc.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'AppStates.dart';
class AppCubit extends Cubit<AppStates> {
  AppCubit():super(AppInitialState());
  static AppCubit get(context)=>BlocProvider.of(context);
 
}