package com.boot.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import com.boot.entity.Team;
import com.boot.entity.User;
import com.boot.service.TeamService;
import com.boot.service.UserService;
import com.boot.vo.BaseMessage;
import com.boot.vo.MessageHandler;
/**
 * team
 */
@Controller
@RequestMapping("/private/team")
public class TeamController {
	@Autowired
	private TeamService teamService;
	@Autowired
	private UserService userService;
	
	@ResponseBody
	@RequestMapping("/list")
	public  BaseMessage<?> teamList() {
		List<Team> list = teamService.getList();
		return MessageHandler.createSuccessVo(list,"查询成功");
	}
	/**
	 * add
	 */
	@RequestMapping("/add")
	public String add(ModelMap model) {
		return "page/team_edit";
	}
	/**
	 * save
	 */
	@ResponseBody
	@RequestMapping("/save")
	public  BaseMessage<?> save(Team team) {
		teamService.save(team);
		return MessageHandler.createSuccessVo("operate successfully");
	}
	@RequestMapping("/addTeamUser")
	public String addTeamUser(ModelMap model,Integer teamId) {
		List<User> userList = userService.getList();
		model.put("userList", userList);
		model.put("teamId", teamId);
		return "page/team_add_user";
	}
	@ResponseBody
	@RequestMapping("/addTeamUser/save")
	public  BaseMessage<?> addTeamUserSave(Integer teamId,Integer userId) {
		teamService.addTeamUser(teamId,userId);
		return MessageHandler.createSuccessVo("operate successfully");
	}
	/**
	 * team 删除 user
	 */
	@ResponseBody
	@RequestMapping("/removeUser")
	public  BaseMessage<?> removeUser(Integer teamId,Integer userId) {
		teamService.removeUser(teamId,userId);
		return MessageHandler.createSuccessVo("operate successfully");
	}
}
