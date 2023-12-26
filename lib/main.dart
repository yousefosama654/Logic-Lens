import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:bloc/bloc.dart';
import 'package:logic_lens/splash_screen/splash_screen.dart';
import 'package:logic_lens/text_screen/text_screen.dart';
import 'cubit/AppCubit.dart';
import 'cubit/AppStates.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MultiBlocProvider(
      // Fix the method name here
      providers: [
        BlocProvider(create: (context) => AppCubit()),
      ],

      child: BlocConsumer<AppCubit, AppStates>(
        listener: (context, states) {},
        builder: (context, states) {
          return MaterialApp(
            debugShowCheckedModeBanner: false,
            title: 'Logic Lens',
            theme: ThemeData(
              primarySwatch: Colors.blue,
            ),
            home:  const SplashScreen(),
          );
        },
      ),
    );
  }
}
