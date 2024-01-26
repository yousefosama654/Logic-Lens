import 'package:flutter/material.dart';
import 'package:logic_lens/helpers.dart';

class TextScreen extends StatelessWidget {
  final String text;
  final String type;
  const TextScreen({Key?key, required this.text, required this.type}): super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Logic Lens'),
        backgroundColor: Colors.blue.withOpacity(0.5),
      ),
      body: SingleChildScrollView(
        scrollDirection: Axis.vertical,
        child: Center(
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                children: [
                  const Text(
                    "Result:",
                    style: TextStyle(
                        fontSize: 23.0,
                        color: Colors.black,
                        fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  type == "Equation"
                      ? buildTable(text)
                      : Text(
                          text,
                          style: const TextStyle(
                              fontSize: 23.0,
                              color: Colors.black,
                              fontWeight: FontWeight.bold),
                        ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
