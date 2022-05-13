package com.boot.controller;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;

import com.boot.entity.Files;
import com.boot.entity.User;
import com.boot.repository.UserRepository;
import com.boot.service.FilesService;
import com.boot.util.FilesUtils;
import com.boot.util.StringUtil;
import com.boot.vo.BaseMessage;
import com.boot.vo.MessageHandler;
import com.boot.vo.Token;
import com.boot.vo.TokenHandler;
/**
 * 登录相关
 */
@Controller
public class LoginController {
	@Autowired
	private UserRepository userRepository;
	@Autowired
	private FilesService filesService;
	
	@GetMapping({"","/login"})
	public String login() {
		return "login";
	}
	
	@ResponseBody
	@PostMapping("/login")
	public BaseMessage<?>  loginP(String username,String password) throws Exception {
		if (StringUtil.isBlank(username) || StringUtil.isBlank(password)) {
			return MessageHandler.createFailedVo("User or password cannot be empty！");
		}
		User user = userRepository.findByUserName(username);
		Map<String, Object> map2 = new HashMap<String, Object>();
		if (user != null) {
			if (password.equals(user.getPassword())) {
				user.setLastTime(new Date());
				userRepository.save(user);
				String token = TokenHandler.create(user.getId(),0,user.getUsername(), user.getName(),null);
				map2.put("token", token);
				map2.put("username", user.getUsername());
				return MessageHandler.createSuccessVo(map2, "login successful");
			} else {
				return MessageHandler.createFailedVo("User password is incorrect");
			}
		} else {
			return MessageHandler.createFailedVo("User does not exist");
		}
	}
	/**
	 * 注册
	 */
	@ResponseBody
	@PostMapping("/register")
	public BaseMessage<?>  register(User user) throws Exception {
		String username = user.getUsername();
		long count = userRepository.countByUserName(username);
		if(count>0)return MessageHandler.createFailedVo("User already exists");
		userRepository.save(user);
		return MessageHandler.createSuccessVo("Register successfully, go to login");
	}
	
	@ResponseBody
	@RequestMapping(value = "/login/checkToken")
	public BaseMessage<?> checkToken(HttpServletRequest request, HttpServletResponse response, ModelMap map) {
		try {
			Token to = TokenHandler.getBusinesser();
			if (to != null) {
				User user = userRepository.findById(to.getBusinessId()).get();
				if (user != null) {
					return MessageHandler.createSuccessVo("token Verified successfully！");
				}
			} else {
				return MessageHandler.createFailedVo("token failure！");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	/**
	 * 获取文件
	 * 图片
	 */
	@RequestMapping("getFile/{name}")
	public void getFile(HttpServletResponse response, @PathVariable("name") String name) {
		try {
			FilesUtils.fileIn(response, name, 2);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	/**
	 * 上传文件
	 */
	@ResponseBody
	@RequestMapping("upload")
	public BaseMessage<?> upload(HttpServletRequest request, MultipartFile file) {
		Map<String,Object> dataMap = new HashMap<>();
		String name = "";
		String id = UUID.randomUUID().toString().replaceAll("-", "");
		try {
			String fileName = file.getOriginalFilename();
			//String fileTyle = fileName.substring(fileName.lastIndexOf("."), fileName.length());
			//name = uuid + fileTyle;
			name = fileName;
			FilesUtils.fileOut(request, file, name);
			Files f = new Files();
			f.setId(id);
			f.setName(name);
			f.setPath(name);
			filesService.save(f);
			dataMap.put("name", name);
			dataMap.put("id", id);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createSuccessVo(dataMap, "operate successfully");
	}
	
	@ResponseBody
	@RequestMapping("download")
	public void download(HttpServletResponse response,HttpServletRequest request,String fileName) {
		System.out.println(fileName);
		try {
			FilesUtils.fileIn(response,fileName,2);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
