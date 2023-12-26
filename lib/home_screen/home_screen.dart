import 'package:flutter/material.dart';
import 'package:bottom_bar_with_sheet/bottom_bar_with_sheet.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:logic_lens/helpers.dart';
import 'package:crea_radio_button/crea_radio_button.dart';
import 'package:logic_lens/text_screen/text_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _selectedItem = "table";
  bool isLoading =false;
  int rad_index = 0;
  final _bottomBarController = BottomBarWithSheetController(initialIndex: 0);
  final ImagePicker picker = ImagePicker();
  XFile? image = null;

  File? file = null;
  @override
  void initState() {
    _bottomBarController.stream.listen((opened) {
      debugPrint('Bottom bar ${opened ? 'opened' : 'closed'}');
    });
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Logic Lens'),
        backgroundColor: Colors.blue.withOpacity(0.5), // Set opacity to 50%
      ),
      bottomNavigationBar: isLoading
          ? null
          : BottomBarWithSheet(
              controller: _bottomBarController,
              bottomBarTheme: const BottomBarTheme(
                mainButtonPosition: MainButtonPosition.middle,
                decoration: BoxDecoration(
                  color: Colors.transparent,
                  borderRadius: BorderRadius.vertical(top: Radius.circular(25)),
                ),
                itemIconColor: Colors.grey,
                itemTextStyle: TextStyle(
                  color: Colors.grey,
                  fontSize: 10.0,
                ),
                selectedItemTextStyle: TextStyle(
                  color: Colors.blue,
                  fontSize: 10.0,
                ),
              ),
              onSelectItem: (index) => debugPrint('$index'),
              sheetChild: ListView(
                children: [
                  Padding(
                    padding: const EdgeInsets.all(48.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Text('Camera',
                            style: TextStyle(
                                fontSize: 20.0,
                                color: Colors.black,
                                fontWeight: FontWeight.bold)),
                        Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Container(
                            width: double.infinity,
                            height: 50.0,
                            color: Colors.blue
                                .withOpacity(0.5), // Set opacity to 50%
                            child: IconButton(
                                onPressed: () async {
                                  image = await picker.pickImage(
                                      source: ImageSource.camera);
                                  if (image != null) {
                                    file = File(image!.path);
                                  }
                                  setState(() {});
                                },
                                icon: const Icon(
                                  Icons.camera_alt_rounded,
                                  size: 28.0,
                                  color: Colors.white,
                                )),
                          ),
                        ),
                        const SizedBox(
                          height: 20.0,
                        ),
                        const Text('Gallery',
                            style: TextStyle(
                                fontSize: 20.0,
                                color: Colors.black,
                                fontWeight: FontWeight.bold)),
                        Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Container(
                            width: double.infinity,
                            height: 50.0,

                            color: Colors.blue
                                .withOpacity(0.5), // Set opacity to 50%
                            child: IconButton(
                              onPressed: () async {
                                image = await picker.pickImage(
                                    source: ImageSource.gallery);
                                if (image != null) {
                                  file = File(image!.path);
                                }
                                setState(() {});
                              },
                              icon: const Icon(
                                Icons.photo,
                                size: 28.0,
                                color: Colors.white,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              items: const [
                BottomBarWithSheetItem(
                  icon: Icons.circle,
                ),
                BottomBarWithSheetItem(icon: Icons.circle),
                BottomBarWithSheetItem(icon: Icons.circle),
                BottomBarWithSheetItem(icon: Icons.circle),
              ],
            ),
      body: isLoading
          ? Center(child: const CircularProgressIndicator())
          : Stack(
              fit: StackFit.expand,
              children: [
                // Background Image
                Image.asset(
                  'assets/back.jpg', // Replace with the path to your image
                  fit: BoxFit.cover,
                ),
                SafeArea(
                  child: ListView(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(58.0),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            Container(
                              child: file == null
                                  ? Image.asset(
                                      'assets/placeholder.png',
                                      fit: BoxFit
                                          .fitWidth, // This will scale the image to cover the entire container while maintaining its aspect ratio.
                                    )
                                  : Image.file(file!),
                            ),
                            const SizedBox(
                              height: 10.0,
                            ),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                              children: [
                                ElevatedButton(
                                  onPressed: () {
                                    file = null;
                                    setState(() {});
                                  },
                                  child: const Text(
                                    'Delete',
                                    style: TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold),
                                  ),
                                ),
                                ElevatedButton(
                                  onPressed: () async {
                                    if (file == null) {
                                      ScaffoldMessenger.of(context)
                                          .showSnackBar(
                                        const SnackBar(
                                          content: Text(
                                            'Please select an image',
                                            style: TextStyle(
                                                color: Colors.white,
                                                fontWeight: FontWeight.bold),
                                          ),
                                          backgroundColor: Colors.red,
                                        ),
                                      );
                                    } else {
                                      setState(() {
                                        isLoading = true;
                                      });
                                      String text = await uploadImage(
                                          file!, _selectedItem);
                                      setState(() {
                                        isLoading = false;
                                      });
                                      Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                          builder: (context) => TextScreen(
                                              text: text, type: _selectedItem),
                                        ),
                                      );
                                    }
                                  },
                                  child: const Text(
                                    'Process',
                                    style: TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold),
                                  ),
                                ),
                              ],
                            ),
                            SafeArea(
                                maintainBottomViewPadding: true,
                                minimum: const EdgeInsets.fromLTRB(0,15,0,0),
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                              children: [
                                RadioButtonGroup(
                                    buttonHeight: 120,
                                    buttonWidth: 120,
                                    circular: true,
                                    mainColor: Colors.grey,
                                    selectedColor: Colors.pink.shade400,
                                    preSelectedIdx: 0,
                                    options: [
                                      RadioOption("Table", "Table"),
                                      RadioOption("Equation", "Equation"),
                                    ],
                                    callback: (RadioOption val) {
                                      setState(() {
                                        _selectedItem = val.value;
                                      });
                                    })
                              ],
                            )),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }
}
