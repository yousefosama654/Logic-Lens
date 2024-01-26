import 'package:flutter/material.dart';
import 'package:another_flutter_splash_screen/another_flutter_splash_screen.dart';
import 'package:logic_lens/home_screen/home_screen.dart';
import 'package:lottie/lottie.dart'; // Import the 'lottie' package

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);


  @override
  State<SplashScreen> createState() => _SplashScreenState();
}
class _SplashScreenState extends State<SplashScreen> {
  @override
  Widget build(BuildContext context) {

        return FlutterSplashScreen(
          useImmersiveMode: true,
          duration: const Duration(milliseconds: 3500),
          nextScreen: HomeScreen(),
          backgroundColor: Colors.white,
          splashScreenBody: Center(
            child: Lottie.asset(
              "assets/Intro.json",
              repeat: false,
            ),
          ),
        );
  }
}